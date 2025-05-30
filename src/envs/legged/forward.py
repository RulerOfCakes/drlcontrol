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

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class LeggedForwardEnv(LeggedEnv):
    """
    Can use any legged robot model, with the forward direction assumed to be +x.

    The goal of this environment is to achieve stable forward locomotion.
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
        xml_file: str = "ant.xml",
        frame_skip: int = 5,  # each 'step' of the environment corresponds to 5 timesteps in the simulation
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        healthy_reward: float = 1.0,  # reward for staying alive
        forward_reward_weight: float = 1.0,  # reward for forward locomotion
        main_body: Union[int, str] = 1,
        termination_contacts: List[Union[int, str]] = [
            1
        ],  # contacts that will terminate the episode
        penalized_contacts: List[
            Union[int, str]
        ] = [],  # contacts that are penalized by collision cost
        termination_height_range: Tuple[float, float] = (
            -np.inf,
            np.inf,
        ),  # height range for termination
        terminate_when_unhealthy: bool = True,  # terminate the episode when the robot is unhealthy
        termination_cost: float = 1000.0,  # penalty for terminating the episode early
        ctrl_cost_weight: float = 0.001,  # penalize large/jerky actions
        collision_cost_weight: float = 0.0001,  # penalize collisions
        reset_noise_scale: float = 0.005,  # noise scale for resetting the robot's position
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        use_circular_terrain_profile: bool = False,
        use_forward_terrain_profile: bool = False,
        circular_terrain_profile_radius: float = 2.0,
        circular_terrain_profile_resolution: int = 5,
        forward_terrain_profile_origin: np.ndarray = np.array([0.5, 0.0, 0.3]),
        forward_terrain_profile_direction: np.ndarray = np.array([1.0, 0.0, 0.0]),
        forward_terrain_profile_length: float = 5.0,
        forward_terrain_profile_dimension: tuple[float, float] = (5, 5),
        forward_terrain_profile_resolution: tuple[int, int] = (5, 5),
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = False,
        include_cvel_in_observation: bool = False,
        include_qfrc_actuator_in_observation: bool = False,
        include_cinert_in_observation: bool = False,
        **kwargs,
    ):
        # initialize the environment variables
        self._healthy_reward = healthy_reward
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._collision_cost_weight = collision_cost_weight

        # healthy - robot must fit in a certain height range(e.g. not falling down)
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._termination_cost = termination_cost

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        # clip range for rewards related to contact forces
        self._contact_force_range = contact_force_range

        self.use_circular_terrain_profile = use_circular_terrain_profile
        self.use_forward_terrain_profile = use_forward_terrain_profile

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = (
            include_qfrc_actuator_in_observation
        )
        self._include_cinert_in_observation = include_cinert_in_observation

        self.metadata = LeggedForwardEnv.metadata

        init_cfg = LeggedInitConfig(
            reset_noise_scale=reset_noise_scale,
        )
        body_cfg = LeggedBodyConfig(
            main_body=main_body,
            termination_contacts=termination_contacts,
            termination_height_range=termination_height_range,
            penalized_contacts=penalized_contacts,
        )
        obs_cfg = LeggedObsConfig(
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
            include_cvel_in_observation=include_cvel_in_observation,
            include_qfrc_actuator_in_observation=include_qfrc_actuator_in_observation,
            include_cinert_in_observation=include_cinert_in_observation,
            include_circular_terrain_profile=use_circular_terrain_profile,
            include_forward_terrain_profile=use_forward_terrain_profile,
            contact_force_range=contact_force_range,
            terrain_profile_circular_radius=circular_terrain_profile_radius,
            terrain_profile_circular_resolution=circular_terrain_profile_resolution,
            terrain_profile_ray_resolution=forward_terrain_profile_resolution,
            terrain_profile_ray_dimension=forward_terrain_profile_dimension,
            terrain_profile_ray_direction=forward_terrain_profile_direction,
            terrain_profile_ray_length=forward_terrain_profile_length,
            terrain_profile_ray_origin=forward_terrain_profile_origin,
        )

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

        # required for MujocoEnv
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.observation_space.shape[0]

        # add a relative angle between the forward vector of the torso and the x-axis
        obs_size += 1

        # metadata for the final observation space
        self.observation_structure["relative_angle"] = 1
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.step_info = {}

    # the z component is ignored as the desired direction is in the x-y plane
    def forward_angle(self):
        target_vector = np.array([-1, 0])

        rot_matrix_body = self.data.xmat[self._main_body]
        rot_matrix_body = rot_matrix_body.reshape(3, 3)

        body_forward_vector = rot_matrix_body @ np.array([-1, 0, 0])
        body_forward_vector = body_forward_vector[:2]

        diff_angle = np.arccos(
            np.clip(
                np.dot(target_vector, body_forward_vector),
                -1.0,
                1.0,
            )
        )
        return diff_angle

    def _get_obs(self):
        # get the current state of the robot
        obs = super()._get_obs()
        angle = np.array([self.forward_angle()])
        obs = np.concatenate([obs, angle])
        return obs

    def _get_rew(self, x_velocity: float, action):
        forward_reward = x_velocity * self._forward_reward_weight
        healthy_reward = self._reward_healthy() * self._healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self._reward_control(action) * self._ctrl_cost_weight
        termination_cost = self.is_terminated * self._termination_cost
        collision_cost = self._reward_collision() * self._collision_cost_weight
        costs = ctrl_cost + termination_cost + collision_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_healthy": healthy_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_termination": -termination_cost,
            "reward_collision": -collision_cost,
        }

        return reward, reward_info

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        x_velocity, y_velocity = (xy_position_after - xy_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = self.is_terminated and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        self.step_info = info.copy()

        return observation, reward, terminated, False, info

    def render(self, *args, **kwargs):
        if self.mujoco_renderer.viewer is None:
            super().render(*args, **kwargs)
        else:
            if (
                self.obs_cfg.include_circular_terrain_profile
                and len(self._prev_circular_profile_coords) > 0
            ):
                self._render_terrain_profile_circular(
                    self._prev_circular_profile_coords
                )
            if (
                self.obs_cfg.include_forward_terrain_profile
                and len(self._prev_forward_profile_coords) > 0
            ):
                self._render_terrain_profile_ray(self._prev_forward_profile_coords)
            self._render_info()
            super().render(*args, **kwargs)

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
            "Forward Reward",
            "Forward Reward: %.6f" % (self.step_info.get("reward_forward") or 0.0),
        )
        self.mujoco_renderer.viewer.add_overlay(
            topright,
            "Control Cost",
            "Control Cost: %.6f" % (self.step_info.get("reward_ctrl") or 0.0),
        )
        self.mujoco_renderer.viewer.add_overlay(
            topright,
            "Collision Cost",
            "Collision Cost: %.6f" % (self.step_info.get("reward_collision") or 0.0),
        )

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
