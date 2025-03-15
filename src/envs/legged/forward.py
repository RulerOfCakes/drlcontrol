from typing import Dict, Tuple, Union
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
import numpy as np

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class LeggedForwardEnv(MujocoEnv):
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
        terminate_when_unhealthy: bool = True,  # terminate the episode when the robot is unhealthy
        termination_cost: float = 1000.0,  # penalty for terminating the episode early
        healthy_z_range: Tuple[float, float] = (
            0.25,
            1.0,
        ),  # z range for the robot to be healthy
        ctrl_cost_weight: float = 0.001,  # penalize large/jerky actions
        contact_cost_weight: float = 0.0005,  # penalize contacts with the ground
        reset_noise_scale: float = 0.005,  # noise scale for resetting the robot's position
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = False,
        **kwargs,
    ):
        # initialize the environment variables
        self._healthy_reward = healthy_reward
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        # healthy - robot must fit in a certain height range(e.g. not falling down)
        self._healthy_z_range = healthy_z_range
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

        self.metadata = LeggedForwardEnv.metadata

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

        # for observations, we collect the model's qpos and qvel
        obs_size = self.data.qpos.size + self.data.qvel.size

        # we also add a relative angle between the forward vector of the torso and the x-axis
        obs_size += 1

        # we may exclude the x and y coordinates of the torso(root link) for position agnostic behavior in policies.
        obs_size -= 2 * exclude_current_positions_from_observation

        # we may include the external forces acting on the robot
        obs_size += len(self.cfrc_ext) * include_cfrc_ext_in_observation

        # metadata for the final observation space
        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
            "direction_angle": 1,
            "cfrc_ext": len(self.cfrc_ext) * include_cfrc_ext_in_observation,
        }

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    @property
    def cfrc_ext(self):
        return self.data.cfrc_ext[1:]

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    @property
    def termination_cost(self):
        return (
            self._terminate_when_unhealthy and (not self.is_healthy)
        ) * self._termination_cost

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

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
        print(diff_angle)
        return diff_angle

    def _get_obs(self):
        # get the current state of the robot
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        angle = np.array([self.forward_angle()])

        # exclude the x and y coordinates of the torso(root link) for position agnostic behavior in policies.
        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        obs = np.concatenate([qpos, qvel, angle])

        # include the external forces acting on the robot
        if self._include_cfrc_ext_in_observation:
            obs = np.concatenate([obs, self.cfrc_ext])

        return obs

    def _get_rew(self, x_velocity: float, action):
        forward_reward = x_velocity * self._forward_reward_weight
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        termination_cost = self.termination_cost
        costs = ctrl_cost + contact_cost + termination_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_termination": -termination_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        x_velocity, y_velocity = (xy_position_after - xy_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
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

        return observation, reward, terminated, False, info

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
