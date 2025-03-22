from dataclasses import dataclass, field
from typing import Dict, List, Union
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
import numpy as np

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


@dataclass
class LeggedInitConfig:
    reset_noise_scale: float = (
        0.005,
    )  # noise scale for resetting the robot's position


@dataclass
class LeggedBodyConfig:
    main_body: Union[int, str] = 1
    termination_contacts: List[Union[int, str]] = field(
        default_factory=list
    )  # body contacts that terminate the episode
    penalized_contacts: List[Union[int, str]] = field(
        default_factory=list
    )  # body contacts that are penalized in the reward function


@dataclass
class LeggedObsConfig:
    exclude_current_positions_from_observation: bool = (
        True  # for position-agnostic behavior
    )
    include_cfrc_ext_in_observation: bool = (
        False  # include external forces in the observation
    )
    contact_force_range: tuple[float, float] = (
        -np.inf,
        np.inf,
    )  # range of contact forces to be clipped in the observation


class LeggedEnv(MujocoEnv):
    """
    Base class containing basic observations and rewards for legged robots.
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
        xml_file: str,
        frame_skip: int = 5,  # each 'step' of the environment corresponds to `frame_skip` timesteps in the simulation
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        init_cfg: LeggedInitConfig = LeggedInitConfig(),
        body_cfg: LeggedBodyConfig = LeggedBodyConfig(),
        obs_cfg: LeggedObsConfig = LeggedObsConfig(),
        **kwargs,
    ):
        self._steps = 0

        self.init_cfg = init_cfg
        self.body_cfg = body_cfg
        self.obs_cfg = obs_cfg

        self.metadata = LeggedEnv.metadata

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # body indices for fast access
        self._termination_contact_indices = [
            (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY)
                if isinstance(name, str)
                else name
            )
            for name in body_cfg.termination_contacts
        ]
        self._penalized_contact_indices = [
            (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY)
                if isinstance(name, str)
                else name
            )
            for name in body_cfg.penalized_contacts
        ]

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

        # we may exclude the x and y coordinates of the torso(root link) for position agnostic behavior in policies.
        obs_size -= 2 * self.obs_cfg.exclude_current_positions_from_observation

        # we may include the external forces acting on the robot
        obs_size += (
            len(self.data.cfrc_ext.ravel())
            * self.obs_cfg.include_cfrc_ext_in_observation
        )

        # metadata for the final observation space
        # both observation_structure and observation_space must be correctly adjusted by child classes
        self.observation_structure = {
            "qpos": self.data.qpos.size
            - 2 * self.obs_cfg.exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cfrc_ext": len(self.data.cfrc_ext.ravel())
            * self.obs_cfg.include_cfrc_ext_in_observation,
        }

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self._prev_action = np.zeros(self.action_space.shape[0], dtype=np.float64)

    @property
    def contact_forces(self):
        """
        Returns a 2D array of contact forces acting on the robot,
        in the order of force x, y, z and torque x, y, z for each body.
        """
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self.obs_cfg.contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def is_terminated(self) -> bool:
        """
        Check if the episode is terminated based on the contacts with the ground.
        """
        # Check for contacts with the ground
        return np.any(
            np.abs(self.contact_forces[self._termination_contact_indices].flatten()) > 0
        )

    ### Rewards
    def _reward_healthy(self) -> float:
        """
        Reward the robot for being healthy.
        """
        return not self.is_terminated

    def _reward_collision(self) -> float:
        """
        Penalize the robot for collisions.
        """
        contact_forces = self.contact_forces
        penalized_contact_forces = np.array(
            [np.abs(arr) for arr in contact_forces[self._penalized_contact_indices]]
        ).flatten()
        return np.sum(penalized_contact_forces)

    def _reward_control(self, action: np.ndarray) -> float:
        """
        Penalize the robot for control actions.
        """
        action_cost = np.sum(np.square(action))
        return action_cost

    def _reward_action_rate(self, prev_action: np.ndarray, action: np.ndarray) -> float:
        """
        Penalize the robot for rapid changes in control actions.
        This is useful for preventing jerky movements.
        """
        action_rate_cost: float = np.sum(np.square(action - prev_action))
        return action_rate_cost

    def _get_obs(self):
        # get the current state of the robot
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # exclude the x and y coordinates of the torso(root link) for position agnostic behavior in policies.
        if self.obs_cfg.exclude_current_positions_from_observation:
            qpos = qpos[2:]

        obs = np.concatenate([qpos, qvel])

        # include the external forces acting on the robot
        if self.obs_cfg.include_cfrc_ext_in_observation:
            obs = np.concatenate([obs, self.data.cfrc_ext])

        return obs

    def _get_rew(self, action):
        reward = -(
            self._reward_collision()
            + self._reward_control(action)
            + self._reward_action_rate(self._prev_action, action)
        )

        reward_info = {
            "reward_collision": -self._reward_collision(),
            "reward_control": -self._reward_control(action),
            "reward_action_rate": -self._reward_action_rate(self._prev_action, action),
        }

        return reward, reward_info

    # This is an example implementation, ideally all child environments should re-implement their own step function
    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        self._steps += 1
        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)

        terminated = self.is_terminated

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def render(self, *args, **kwargs):
        super().render(*args, **kwargs)

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

        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation
