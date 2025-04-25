from typing import Dict, List, Tuple, Union
from gymnasium.spaces import Box
import mujoco
import numpy as np

from envs.legged.base import (
    LeggedBodyConfig,
    LeggedEnv,
    LeggedInitConfig,
    LeggedObsConfig,
)
from utils.linalg import rotation_matrix

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class LeggedTerrainEnv(LeggedEnv):
    """
    Can use any legged robot model.

    The goal of this environment is to achieve stable locomotion on bumpy terrain.
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
        xml_file: str = "boston_dynamics_spot/scene_terrain.xml",
        frame_skip: int = 5,  # each 'step' of the environment corresponds to 5 timesteps in the simulation
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 10.0,  # reward for getting closer to the target
        angular_reward_weight: float = 1.0,  # reward for aligning the robot's forward direction with the target direction
        success_reward_weight: float = 1000.0,  # reward for reaching the target
        termination_cost: float = 1000.0,  # penalty for terminating the episode early
        ctrl_cost_weight: float = 0.0001,  # penalize large/jerky actions
        action_rate_cost_weight: float = 0.0001,  # penalize large/jerky actions
        main_body: Union[int, str] = 1,
        terminate_when_unhealthy: bool = True,  # terminate the episode when the robot is unhealthy
        termination_contacts: List[Union[int, str]] = [1],
        reset_noise_scale: float = 0.005,  # noise scale for resetting the robot's position
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        terrain_profile_radius: float = 2.0,  # radius of the terrain profile around the robot
        terrain_profile_resolution: int = 5,  # resolution of the terrain profile
        spawn_radius: float = 5.0,  # radius of the spawn area for the robot
        randomize_target_position: bool = True,
        initial_target_range: float = 4.0,
        initial_target_angular_range: float = 0.0,
        max_target_angular_range: float = np.pi,
        max_target_range=10.0,  # absolute coordinate(x, y) range for the target position
        target_range_increment: float = 0.1,
        target_angular_range_increment: float = 0.05,
        increment_frequency: int = 5000,  # Environment steps per increment (# of calls to `step()`)
        **kwargs,
    ):
        # initialize the reward variables
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._success_reward_weight = success_reward_weight
        self._angular_reward_weight = angular_reward_weight
        self._termination_cost = termination_cost
        self._action_rate_cost_weight = action_rate_cost_weight

        self._terminate_when_unhealthy = terminate_when_unhealthy

        # configuration for base LeggedEnv
        init_cfg = LeggedInitConfig(
            reset_noise_scale=reset_noise_scale,
        )
        body_cfg = LeggedBodyConfig(
            main_body=main_body,
            termination_contacts=termination_contacts,
            penalized_contacts=[],
        )
        obs_cfg = LeggedObsConfig(
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
            contact_force_range=contact_force_range,
            terrain_profile_circular_radius=terrain_profile_radius,
            terrain_profile_circular_resolution=terrain_profile_resolution,
        )

        self._max_target_range = max_target_range
        self._target_range = initial_target_range
        self._target_angular_range = initial_target_angular_range
        self._max_target_angular_range = max_target_angular_range
        self._target_range_increment = target_range_increment
        self._target_angular_range_increment = target_angular_range_increment
        self._increment_frequency = increment_frequency
        self._steps = 0

        self._spawn_radius = spawn_radius

        self._randomize_target_position = randomize_target_position

        self.metadata = LeggedTerrainEnv.metadata

        LeggedEnv.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config=default_camera_config,
            init_cfg=init_cfg,
            body_cfg=body_cfg,
            obs_cfg=obs_cfg,
            **kwargs,
        )

        self.target_pos = self._generate_target_position(
            self.data.qpos[0], self.data.qpos[1]
        )

        # load observation size from parent class
        obs_size = self.observation_space.shape[0]

        # relative target position vector (x,y)
        obs_size += 2

        # relative velocity vector (x,y,z)
        obs_size += 3

        # terrain profile
        obs_size += terrain_profile_resolution**2

        # modify metadata for the final observation space
        self.observation_structure["relative_target_pos"] = 2
        self.observation_structure["relative_velocity"] = 3
        self.observation_structure["terrain_profile"] = terrain_profile_resolution**2

        # update the observation space
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self._prev_pos = self.data.body(
            self.body_cfg.main_body
        ).xpos.copy()  # previous position of the robot
        self._last_profile_coords = np.array([])
        self.step_info = {}  # for debugging purposes

    @property
    def goal_reached(self) -> bool:
        distance_to_target = np.linalg.norm(
            self.target_pos - self.data.body(self.body_cfg.main_body).xpos[:2]
        )
        return distance_to_target < 0.5

    ## Rewards
    def _reward_termination(self):
        return self._terminate_when_unhealthy and (self.is_terminated)

    def _reward_success(self):
        return self.goal_reached

    def _reward_angular(self):
        body_rotation = self.data.body(self.body_cfg.main_body).xmat.reshape(3, 3)

        # Extract the forward direction of the robot
        forward_direction = body_rotation[:, 0]  # Assuming forward is along the x-axis
        forward_direction = (
            forward_direction / np.linalg.norm(forward_direction)
            if np.linalg.norm(forward_direction) > 0
            else np.zeros_like(forward_direction)
        )
        # omit the z coordinate
        forward_direction = forward_direction[:2]

        # Calculate the target direction
        target_direction = (
            self.target_pos - self.data.body(self.body_cfg.main_body).xpos[:2]
        )
        target_direction = (
            target_direction / np.linalg.norm(target_direction)
            if np.linalg.norm(target_direction) > 0
            else np.zeros_like(target_direction)
        )

        dot = np.dot(forward_direction, target_direction)
        sign = np.sign(dot)

        return np.square(dot) * sign  # reward for aligning with the target direction

    def _reward_forward(self):
        current_pos = self.data.body(self.body_cfg.main_body).xpos[:2].copy()
        displacement = current_pos - self._prev_pos[:2]
        target_direction = self.target_pos - current_pos
        target_direction /= np.linalg.norm(target_direction)

        # reward for moving in the direction of the target
        displacement_on_target_direction = np.dot(displacement, target_direction)
        return displacement_on_target_direction

    # z coordinate is omitted as the robot is expected to move in the x-y plane
    def _generate_target_position(self, center_x: float, center_y: float):
        r = self.np_random.uniform(low=self._target_range, high=self._target_range)
        theta = self.np_random.uniform(
            low=-self._target_angular_range,
            high=self._target_angular_range,
        )
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.array([x + center_x, y + center_y], dtype=np.float32)

    def _get_obs(self):
        # get the current state of the robot
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        body_rot: np.ndarray = (
            self.data.body(self.body_cfg.main_body).xmat.copy().reshape(3, 3)
        )

        # the linear velocities of the robot's body in qvel are in the global frame
        # we need to convert them to the local frame of the robot's body
        body_lin_vel = qvel[:3].copy()
        body_lin_vel = (
            body_rot.T @ body_lin_vel
        )  # Transpose is required to convert from world frame to local frame
        qvel[:3] = body_lin_vel

        # compute the relative target position in the robot's local frame
        relative_target_pos: np.ndarray = (
            self.target_pos - self.data.body(self.body_cfg.main_body).xpos[:2]
        )
        relative_target_pos = np.append(
            relative_target_pos, 0.0
        )  # add a z coordinate for the rotation matrix
        relative_target_pos = body_rot.T @ relative_target_pos
        relative_target_pos = relative_target_pos[:2]  # exclude the z coordinate

        body_vel = (
            self.data.body(self.body_cfg.main_body).xpos - self._prev_pos
        ) / self.dt
        # compute the relative direction of velocity in the same way
        relative_vel: np.ndarray = body_rot.T @ body_vel

        # exclude the x and y coordinates of the torso(root link) for position agnostic behavior in policies.
        if self.obs_cfg.exclude_current_positions_from_observation:
            qpos = qpos[2:]

        profile_coords, profile_data = self.terrain_profile_circular()

        # save data for rendering phase
        self._last_profile_coords = profile_coords

        obs = np.concatenate(
            [qpos, qvel, relative_target_pos, relative_vel, profile_data]
        )

        # include the external forces acting on the robot
        if self.obs_cfg.include_cfrc_ext_in_observation:
            obs = np.concatenate([obs, self.contact_forces.flatten()])

        return obs

    def _get_rew(self, action: np.ndarray):
        forward_reward = self._reward_forward() * self._forward_reward_weight
        angular_reward = self._reward_angular() * self._angular_reward_weight
        success_reward = self._reward_success() * self._success_reward_weight
        rewards = forward_reward + success_reward + angular_reward

        ctrl_cost = self._reward_control(action) * self._ctrl_cost_weight
        action_rate_cost = (
            self._reward_action_rate(self._prev_action, action)
            * self._action_rate_cost_weight
        )
        termination_cost = self._reward_termination() * self._termination_cost
        costs = ctrl_cost + termination_cost + action_rate_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_angular": angular_reward,
            "reward_success": success_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_termination": -termination_cost,
            "reward_action_rate": -action_rate_cost,
        }

        self._prev_pos = self.data.body(self.body_cfg.main_body).xpos.copy()
        self._prev_action = action.copy()

        return reward, reward_info

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        self._steps += 1

        if self._steps % self._increment_frequency == 0:
            self._target_range = min(
                self._target_range + self._target_range_increment,
                self._max_target_range,
            )
            self._target_angular_range = min(
                self._target_angular_range + self._target_angular_range_increment,
                self._max_target_angular_range,
            )

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)

        # termination condition 1. unhealthy
        terminated = (self.is_terminated) and self._terminate_when_unhealthy
        # termination condition 2. goal reached
        terminated = terminated or self.goal_reached

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "target_position": self.target_pos,
            **reward_info,
        }
        self.step_info = info.copy()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def render(self, *args, **kwargs):
        if self.mujoco_renderer.viewer is None:
            super().render(*args, **kwargs)
        else:
            self._render_target()
            self._render_info()
            if len(self._last_profile_coords) > 0:
                self._render_terrain_profile_circular(self._last_profile_coords)
            super().render(*args, **kwargs)

    def _render_target(self):
        self._render_site(
            np.concatenate(
                [self.target_pos, [self._get_ground_height(*self.target_pos)]]
            )
        )

    def _render_forward_dir(self):
        # Render the target position in the forward direction of the robot
        body_rotation = self.data.body(self.body_cfg.main_body).xmat.reshape(3, 3)
        # forward_direction = body_rotation @ np.array([1, 0, 0])  # Assuming forward is along the x-axis
        forward_direction = body_rotation[:, 0]
        forward_direction = (
            forward_direction / np.linalg.norm(forward_direction)
            if np.linalg.norm(forward_direction) > 0
            else np.zeros_like(forward_direction)
        )

        self._render_arrow(
            origin=self.data.body(self.body_cfg.main_body).xpos.copy(),
            dir=forward_direction,
            length=1.0,
        )

    def _render_info(self):
        if self.render_mode != "human":
            return
        topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        self.mujoco_renderer.viewer.add_overlay(
            topright,
            "Current Position",
            "x: %.2f, y: %.2f"
            % (
                self.data.body(self.body_cfg.main_body).xpos[0],
                self.data.body(self.body_cfg.main_body).xpos[1],
            ),
        )
        self.mujoco_renderer.viewer.add_overlay(
            topright,
            "Target Position",
            "x: %.2f, y: %.2f" % (self.target_pos[0], self.target_pos[1]),
        )
        self.mujoco_renderer.viewer.add_overlay(
            topright,
            "Forward Reward",
            "Forward Reward: %.6f" % (self.step_info.get("reward_forward") or 0.0),
        )
        self.mujoco_renderer.viewer.add_overlay(
            topright,
            "Angular Reward",
            "Angular Reward: %.6f" % (self.step_info.get("reward_angular") or 0.0),
        )
        self.mujoco_renderer.viewer.add_overlay(
            topright,
            "Action Rate Cost",
            "Action Rate Cost: %.6f"
            % (self.step_info.get("reward_action_rate") or 0.0),
        )
        self.mujoco_renderer.viewer.add_overlay(
            topright,
            "Control Cost",
            "Control Cost: %.6f" % (self.step_info.get("reward_ctrl") or 0.0),
        )

    def reset_model(self):
        noise_low = -self.init_cfg.reset_noise_scale
        noise_high = self.init_cfg.reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self.init_cfg.reset_noise_scale
            * self.np_random.standard_normal(self.model.nv)
        )

        # random spawn position
        radius = np.random.uniform(0, self._spawn_radius)
        theta = np.random.uniform(0, 2 * np.pi)

        spawn_x = radius * np.cos(theta)
        spawn_y = radius * np.sin(theta)

        qpos[0] = spawn_x
        qpos[1] = spawn_y

        # Ensure that the starting position is above the ground
        # TODO: extract information from the heightfield
        ground_height = 1.3
        qpos[2] = max(qpos[2], ground_height)

        if self._randomize_target_position:
            self.target_pos = self._generate_target_position(spawn_x, spawn_y)

        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation
