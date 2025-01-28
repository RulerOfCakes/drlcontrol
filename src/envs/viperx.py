from typing import Dict, Optional, Tuple, Union
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
import mujoco

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class ViperXEnv(MujocoEnv):
    """
    ViperXEnv uses the Trossen ViperX 300s robot arm to grab objects and move them in a scene.

    On initialization, a box will be spawned at a random location in the scene, reachable by the robot arm.
    The robot arm should then grab the box to move it towards a target location.

    Args:

        xml_file (str): The path to the xml file to load the environment from.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "trossen_vx300s/scene_box.xml",
        frame_skip: int = 5,  # each 'step' of the environment corresponds to 5 timesteps in the simulation
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        gripper_distance_reward_weight: float = 0.5,
        box_distance_reward_weight: float = 1,
        grasp_reward_weight: float = 2,
        success_reward: float = 100,
        time_penalty_weight: float = 0.0001,  # penalize long episodes
        collision_penalty_weight: float = 0.0,  # penalize collisions - set to 0 for now
        ctrl_cost_weight: float = 0.1,  # penalize large/jerky actions
        goal_tolerance: float = 0.015,  # tolerated distance from goal position to be considered as success
        reset_noise_scale: float = 0.005,  # noise scale for resetting the robot's position
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        include_cfrc_ext_in_observation: bool = False,
        randomize_box_position: bool = False,
        **kwargs,
    ):
        # initialize the environment variables
        self._gripper_distance_reward_weight = gripper_distance_reward_weight
        self._box_distance_reward_weight = box_distance_reward_weight
        self._grasp_reward_weight = grasp_reward_weight
        self._success_reward = success_reward
        self._time_penalty_weight = time_penalty_weight
        self._collision_penalty_weight = collision_penalty_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._goal_tolerance = goal_tolerance
        self._reset_noise_scale = reset_noise_scale

        # clip range for rewards related to contact forces
        self._contact_force_range = contact_force_range

        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation
        self._randomize_box_position = randomize_box_position

        self.metadata = ViperXEnv.metadata

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # required for MujocoEnv
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self._reset_goal(np.array([0.0, 0.4, 0.025]))

        # for observations, we collect the model's qpos and qvel, and the goal position
        obs_size = self.data.qpos.size + self.data.qvel.size + 3
        # we may include the external forces acting on the gripper
        obs_size += len(self.gripper_cfrc_ext()) * include_cfrc_ext_in_observation

        # metadata for the final observation space
        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
            "goal_position": 3,
            "cfrc_ext": len(self.gripper_cfrc_ext()) * include_cfrc_ext_in_observation,
        }

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def _generate_valid_location(self) -> np.ndarray:
        spawn_height = 0.025
        spawn_radius_lower = 0.3
        spawn_radius_upper = 0.5

        r = np.random.uniform(spawn_radius_lower, spawn_radius_upper)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = spawn_height

        return np.array([x, y, z])

    def _get_box_position(self):
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        return self.data.xpos[box_id]

    def _spawn_box(self, spawn_position: Optional[np.ndarray] = None):
        """
        Spawns the box at the given position or at a random position near the origin.

        Parameters:
        - spawn_position: Optional np.array of shape (3,) specifying x, y, z coordinates.
        """

        if spawn_position is None:
            # Randomly generate spawn position within the limits
            spawn_position = self._generate_valid_location()

        else:
            spawn_position = np.array(spawn_position)

        self.spawn_position = spawn_position

        # Get the index of the box's free joint in qpos
        box_joint_name = "box_free_joint"
        box_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, box_joint_name
        )

        # Get the starting index of the box's free joint in qpos
        box_qpos_adr = self.model.jnt_qposadr[box_joint_id]

        # Set the position and orientation in qpos
        # For a free joint, qpos has 7 elements: [x, y, z, qw, qx, qy, qz]
        # We'll set orientation to no rotation (identity quaternion)
        self.data.qpos[box_qpos_adr : box_qpos_adr + 3] = spawn_position  # Set position
        self.data.qpos[box_qpos_adr + 3 : box_qpos_adr + 7] = np.array(
            [1, 0, 0, 0]
        )  # Set orientation (w, x, y, z)

        # Reset the box's velocities to zero
        box_qvel_adr = self.model.jnt_dofadr[box_joint_id]
        self.data.qvel[box_qvel_adr : box_qvel_adr + 6] = np.zeros(6)

        # Recompute positions and velocities
        mujoco.mj_forward(self.model, self.data)

    def _get_child_body_ids(self, body_id):
        ids = []
        child_body_ids = np.where(self.model.body_parentid == body_id)[0]
        for child_body_id in child_body_ids:
            ids.append(child_body_id)
            ids.extend(self._get_child_body_ids(child_body_id))
        return ids

    @property
    def gripper_body_ids(self):
        # the ids of the bodies that are part of the gripper
        gripper_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link"
        )
        return [gripper_body_id] + self._get_child_body_ids(gripper_body_id)

    @property
    def gripper_geom_ids(self):
        # the ids of the geoms that are part of the gripper
        gripper_body_ids = self.gripper_body_ids
        mask = np.isin(self.model.geom_bodyid, gripper_body_ids)
        return np.nonzero(mask)[0]

    @property
    def box_geom_ids(self):
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        return np.array([self.model.body_geomadr[box_id]])

    def gripper_cfrc_ext(self):
        # We only care about the contact forces acting on the gripper
        raw_contact_forces = self.data.cfrc_ext[self.gripper_body_ids]
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    def is_grasping_object(self, object_geom_ids, gripper_geom_ids):
        """
        A loose sense of 'grasping' is defined as the gripper and the object having a contact point.
        """
        for contact in self.data.contact:
            geom1, geom2 = contact.geom1, contact.geom2
            if geom1 in object_geom_ids and geom2 in gripper_geom_ids:
                return True
            if geom2 in object_geom_ids and geom1 in gripper_geom_ids:
                return True
        return False

    def _is_box_at_goal(self):
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_position = self.data.xpos[box_id]

        return np.linalg.norm(box_position - self.goal_position) < self._goal_tolerance

    def _store_previous_state(self):
        box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")

        box_pos = self.data.xpos[box_body_id]
        box_quat = self.data.xquat[box_body_id]
        self.previous_state = {
            "box_position": box_pos,
            "box_orientation": box_quat,
        }

    # Reward 0: Gripper -> Box Distance Reward
    # This motivates the robot arm to move the gripper closer to the box
    @property
    def gripper_distance_reward(self):
        gripper_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link"
        )
        gripper_position = self.data.xpos[gripper_id]

        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_position = self.data.xpos[box_id]

        # Scheme 1. reward the reduction in distance between the gripper and the box
        # Get current and previous positions of the gripper
        # current_distance = np.linalg.norm(gripper_position - box_position)
        # previous_distance = np.linalg.norm(
        #     self.previous_state["box_position"] - box_position
        # )

        # Calculate reward based on reduction in distance
        # distance_reward = self._gripper_distance_reward_weight * (
        #     previous_distance - current_distance
        # )

        # Scheme 2. penalize the distance between the gripper and the box
        distance_reward = -self._gripper_distance_reward_weight * np.linalg.norm(
            gripper_position - box_position
        )

        return distance_reward

    # Reward 1: Box -> Goal Distance Reward
    # This motivates the robot arm to move the box closer to the goal
    @property
    def box_distance_reward(self):
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_position = self.data.xpos[box_id]

        # Scheme 1. reward the reduction in distance between the box and the goal
        # Get current and previous positions of the box
        # current_distance = np.linalg.norm(box_position - self.goal_position)
        # previous_distance = np.linalg.norm(
        #     self.previous_state["box_position"] - self.goal_position
        # )

        # Calculate reward based on reduction in distance
        # distance_reward = self._box_distance_reward_weight * (
        #     previous_distance - current_distance
        # )

        # Scheme 2. penalize the distance between the box and the goal
        distance_reward = -self._box_distance_reward_weight * np.linalg.norm(
            box_position - self.goal_position
        )

        return distance_reward

    # Reward 2: Grasp Reward
    # This motivates the robot arm to move closer to the box and grasp it
    @property
    def grasp_reward(self):
        box_geom_ids = self.box_geom_ids
        gripper_geom_ids = self.gripper_geom_ids

        grasp_cost = self._grasp_reward_weight * self.is_grasping_object(
            box_geom_ids, gripper_geom_ids
        )
        return grasp_cost


    # Reward 3: Success Reward
    # Note that once the success reward was given, the episode will be terminated
    @property
    def success_reward(self):
        if self._is_box_at_goal():
            return self._success_reward
        else:
            return 0

    # Reward 4: Time Penalty
    @property
    def time_penalty_reward(self):
        penalty = -self.data.time * self._time_penalty_weight
        return penalty

    # Reward 5: Collision Penalty
    # Note that usually the grasp reward should be set high enough to outweigh the collision penalty
    @property
    def collision_penalty_reward(self):
        penalty = -self.data.ncon * self._collision_penalty_weight
        return penalty

    # Reward 6: Control Cost
    def control_penalty_reward(self, action):
        control_cost = -self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def distance_from_box(self):
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_position = self.data.xpos[box_id]
        return np.linalg.norm(box_position - self.goal_position)

    def step(self, action):
        self._store_previous_state()
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        terminated = self._is_box_at_goal()

        info = {
            "distance_from_box": self.distance_from_box,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_rew(self, action):
        gripper_distance_reward = self.gripper_distance_reward
        box_distance_reward = self.box_distance_reward
        grasp_reward = self.grasp_reward
        success_reward = self.success_reward

        # the penalty rewards are negative in nature
        time_penalty = self.time_penalty_reward
        collision_penalty = self.collision_penalty_reward
        control_penalty = self.control_penalty_reward(action)

        total_reward = (
            gripper_distance_reward
            + box_distance_reward
            + grasp_reward
            + success_reward
            + time_penalty
            + collision_penalty
            + control_penalty
        )

        reward_info = {
            "gripper_distance_reward": gripper_distance_reward,
            "box_distance_reward": box_distance_reward,
            "grasp_reward": grasp_reward,
            "success_reward": success_reward,
            "time_penalty": time_penalty,
            "collision_penalty": collision_penalty,
            "control_penalty": control_penalty,
            "total_reward": total_reward,
        }

        return total_reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._include_cfrc_ext_in_observation:
            contact_force = self.gripper_cfrc_ext().flatten()
            return np.concatenate(
                (position, velocity, self.goal_position, contact_force)
            )
        else:
            return np.concatenate((position, velocity, self.goal_position))

    def _reset_goal(self, spawn_position: Optional[np.ndarray] = None):
        goal_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site"
        )
        if spawn_position is not None:
            self.goal_position = np.array(spawn_position)
        else:
            self.goal_position = self._generate_valid_location()

        self.model.site_pos[goal_site_id] = self.goal_position

        mujoco.mj_forward(self.model, self.data)

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        # TODO: check if this resets data.time
        self.set_state(qpos, qvel)

        if self._randomize_box_position:
            self._spawn_box()
            self._reset_goal()
        else:
            self._spawn_box(self._get_box_position())
            self._reset_goal(self.goal_position)

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {"distance_from_box": self.distance_from_box}

    # def _get_info(self):
    #     return {
    #         "distance": np.linalg.norm(
    #             self._agent_location - self._target_location, ord=1
    #         )
    #     }
