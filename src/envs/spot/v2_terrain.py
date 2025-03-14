from typing import Dict, List, Tuple, Union
from gymnasium.spaces import Box
import mujoco
import numpy as np

from envs.legged import LeggedBodyConfig, LeggedEnv, LeggedInitConfig, LeggedObsConfig

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class SpotEnvV2(LeggedEnv):
    """
    Uses the Spot Quadruped developed by Boston Dynamics.

    The forward direction is assumed to be +x.

    The goal of this environment is to achieve stable forward locomotion on fluctuous terrain.

    Early termination will be detected by external contacts on the main body.
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
        xml_file: str = "boston_dynamics_spot/scene_v2.xml",
        frame_skip: int = 5,  # each 'step' of the environment corresponds to 5 timesteps in the simulation
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 2,  # reward for forward locomotion
        main_body: Union[int, str] = 1,
        terminate_when_unhealthy: bool = True,  # terminate the episode when the robot is unhealthy
        termination_cost: float = 1000.0,  # penalty for terminating the episode early
        ctrl_cost_weight: float = 0.1,  # penalize large/jerky actions
        action_rate_cost_weight: float = 0.1,  # penalize actions that greatly differ from the previous action
        terrain_profile_radius: float = 2.0,  # radius of the terrain profile around the robot
        terrain_profile_resolution: int = 5,  # resolution of the terrain profile
        reset_noise_scale: float = 0.005,  # noise scale for resetting the robot's position
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        **kwargs,
    ):
        # initialize the environment variables
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._action_rate_cost_weight = action_rate_cost_weight

        # initialize termination contacts
        termination_contacts = [main_body]
        # TODO: initialize penalized contacts
        penalized_contacts = []

        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._termination_cost = termination_cost

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        # clip range for rewards related to contact forces
        self._contact_force_range = contact_force_range

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        self._terrain_profile_radius = terrain_profile_radius
        self._terrain_profile_resolution = terrain_profile_resolution

        self.metadata = SpotEnvV2.metadata

        LeggedEnv.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            init_cfg=LeggedInitConfig(reset_noise_scale=reset_noise_scale),
            body_cfg=LeggedBodyConfig(
                main_body=main_body,
                termination_contacts=termination_contacts,
                penalized_contacts=penalized_contacts,
            ),
            obs_cfg=LeggedObsConfig(
                exclude_current_positions_from_observation=exclude_current_positions_from_observation,
                include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
                contact_force_range=contact_force_range,
            ),
            **kwargs,
        )

        # Additional observations aside from LeggedEnv may include the nearby terrain profile

        obs_size: int = self.observation_space.shape[0]
        obs_size += terrain_profile_resolution**2

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float64,
        )
        self.observation_structure["terrain_profile"] = terrain_profile_resolution**2

        self._prev_pos = self.data.qpos[:3].copy()

        self.terrain_profile_site_ids: List[Union[int, str]] | None = None

    @property
    def termination_cost(self):
        return (
            self._terminate_when_unhealthy and self.is_terminated
        ) * self._termination_cost

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def terrain_profile(self) -> np.ndarray:
        """
        Returns the terrain profile around the robot.
        """
        rad = self._terrain_profile_radius
        res = self._terrain_profile_resolution

        robot_pos = self.data.qpos[:3]  # get the x and y coordinates of the robot
        robot_x, robot_y, robot_z = robot_pos[0], robot_pos[1], robot_pos[2]

        profile_coords = [
            (x, y)
            for x in np.linspace(robot_x - rad, robot_x + rad, res)
            for y in np.linspace(robot_y - rad, robot_y + rad, res)
        ]

        terrain_profile_list = []
        for world_x, world_y in profile_coords:
            world_point = np.array([world_x, world_y, 0.0], dtype=np.float64)
            ray_start = (world_point + np.array([0, 0, robot_z])).reshape(
                3, 1
            )  # assuming that the main body is at the center of the robot above ground
            ray_dir = (world_point + np.array([0, 0, -1])).reshape(3, 1)
            intersection_geoms = np.zeros((1, 1), dtype=np.int32)

            result = mujoco.mj_ray(
                self.model,
                self.data,
                ray_start,
                ray_dir,
                None,
                1,  # include static geoms
                self._main_body,
                intersection_geoms,
            )

            if result != -1:
                intersection_distance_from_body_height = result
                terrain_profile_list.append(intersection_distance_from_body_height)
            else:
                terrain_profile_list.append(0.0)  # default value can be changed

        terrain_profile_array = np.array(terrain_profile_list, dtype=np.float64)
        return terrain_profile_array

    def _get_obs(self):
        # get the current state of the robot
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # exclude the x and y coordinates of the torso(root link) for position agnostic behavior in policies.
        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        obs = np.concatenate([qpos, qvel])

        # include the terrain profile around the robot
        obs = np.concatenate([obs, self.terrain_profile()])

        # include the external forces acting on the robot
        if self._include_cfrc_ext_in_observation:
            obs = np.concatenate([obs, self.contact_forces.flatten()])

        return obs

    def _get_rew(self, action):
        forward_reward = self._forward_reward_weight * (
            self.data.qpos[0] - self._prev_pos[0]
        )
        rewards = forward_reward

        # penalize moving away from the +x axis
        sway_cost = np.abs(self.data.qpos[1])
        ctrl_cost = self.control_cost(action)
        termination_cost = self.termination_cost
        action_rate_cost = (
            self._reward_action_rate(self._prev_action, action)
            * self._action_rate_cost_weight
        )
        costs = sway_cost + ctrl_cost + termination_cost + action_rate_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_sway": -sway_cost,
            "reward_ctrl": -ctrl_cost,
            "reward_termination": -termination_cost,
            "reward_action_rate": -action_rate_cost,
        }

        self._prev_action = action
        self._prev_pos = self.data.qpos[:3].copy()

        return reward, reward_info

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        self._steps += 1

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)

        # termination condition 1. unhealthy
        terminated = self.is_terminated and self._terminate_when_unhealthy

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def render_terrain_profile(self):
        if self.render_mode != "human":
            return

        # get the terrain profile around the robot

    def render(self, *args, **kwargs):
        super().render(*args, **kwargs)

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

        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation
