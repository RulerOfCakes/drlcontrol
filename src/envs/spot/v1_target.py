from typing import Dict, Tuple, Union
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
import numpy as np

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class SpotEnvV1(MujocoEnv):
    """
    Uses the Spot Quadruped developed by Boston Dynamics.

    The goal of this environment is to achieve stable locomotion towards a given target position.
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
        xml_file: str = "boston_dynamics_spot/scene_v1.xml",
        frame_skip: int = 5,  # each 'step' of the environment corresponds to 5 timesteps in the simulation
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 10.0,  # reward for getting closer to the target
        success_reward_weight: float = 10000.0,  # reward for reaching the target
        main_body: Union[int, str] = 1,
        terminate_when_unhealthy: bool = True,  # terminate the episode when the robot is unhealthy
        termination_cost: float = 10000.0,  # penalty for terminating the episode early
        healthy_z_range: Tuple[float, float] = (
            0.25,
            1.0,
        ),  # z range for the robot to be healthy
        ctrl_cost_weight: float = 0.1,  # penalize large/jerky actions
        reset_noise_scale: float = 0.005,  # noise scale for resetting the robot's position
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = False,
        max_target_range=5.0,  # absolute coordinate(x, y) range for the target position
        randomize_target_position: bool = True,
        initial_target_range: float = 1.0,
        target_range_increment: float = 0.1,
        increment_frequency: int = 1000,  # Environment steps per increment (# of calls to `step()`)
        **kwargs,
    ):
        # initialize the environment variables
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._success_reward_weight = success_reward_weight

        # healthy - robot must fit in a certain height range(e.g. not falling down)
        self._healthy_z_range = healthy_z_range
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._termination_cost = termination_cost

        self._max_target_range = max_target_range
        self._target_range = initial_target_range
        self._target_range_increment = target_range_increment
        self._increment_frequency = increment_frequency
        self._steps = 0

        self._randomize_target_position = randomize_target_position
        self.target_pos = self._generate_target_position()

        self.target_site_id = None  # site ID for the target position

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        # clip range for rewards related to contact forces
        self._contact_force_range = contact_force_range

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        self.metadata = SpotEnvV1.metadata

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

        # we also add a relative target position vector (x,y)
        obs_size += 2

        # we may exclude the x and y coordinates of the torso(root link) for position agnostic behavior in policies.
        obs_size -= 2 * exclude_current_positions_from_observation

        # we may include the external forces acting on the robot
        obs_size += len(self.cfrc_ext) * include_cfrc_ext_in_observation

        # metadata for the final observation space
        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
            "relative_target_pos": 2,
            "cfrc_ext": len(self.cfrc_ext) * include_cfrc_ext_in_observation,
        }

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    @property
    def cfrc_ext(self):
        return self.data.cfrc_ext[1:]

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
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def goal_reached(self) -> bool:
        distance_to_target = np.linalg.norm(
            self.target_pos - self.data.body(self._main_body).xpos[:2]
        )
        return distance_to_target < 0.2

    @property
    def success_reward(self):
        return self.goal_reached * self._success_reward_weight

    # z coordinate is omitted as the robot is expected to move in the x-y plane
    def _generate_target_position(self):
        x = self.np_random.uniform(low=-self._target_range, high=self._target_range)
        y = self.np_random.uniform(low=-self._target_range, high=self._target_range)
        return np.array([x, y], dtype=np.float32)

    def _get_obs(self):
        # get the current state of the robot
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        relative_target_pos = self.target_pos - self.data.body(self._main_body).xpos[:2]

        # exclude the x and y coordinates of the torso(root link) for position agnostic behavior in policies.
        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        obs = np.concatenate([qpos, qvel, relative_target_pos])

        # include the external forces acting on the robot
        if self._include_cfrc_ext_in_observation:
            obs = np.concatenate([obs, self.cfrc_ext])

        return obs

    def _get_rew(self, action):
        distance_to_target = np.linalg.norm(
            self.target_pos - self.data.body(self._main_body).xpos[:2]
        )

        forward_reward = self._forward_reward_weight * -distance_to_target

        success_reward = self.success_reward
        rewards = forward_reward + success_reward

        ctrl_cost = self.control_cost(action)
        termination_cost = self.termination_cost
        costs = ctrl_cost + termination_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_success": success_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_termination": -termination_cost,
        }

        return reward, reward_info

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        self._steps += 1
        if self._steps % self._increment_frequency == 0:
            self._target_range = min(
                self._target_range + self._target_range_increment,
                self._max_target_range,
            )

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)

        # termination condition 1. unhealthy
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        # termination condition 2. goal reached
        terminated = terminated or self.goal_reached

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
        self._render_target()
        super().render(*args, **kwargs)

    def _render_target(self):
        if self.target_site_id is None:
            # Find the site ID (or create one in your XML if it doesn't exist)
            self.target_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site"
            )
            if self.target_site_id == -1:
                raise ValueError(
                    "Site 'target_site' not found in the XML model. Please add it."
                )

        # Update the site's position
        self.model.site_pos[self.target_site_id] = np.concatenate(
            [self.target_pos, [0.1]]
        )  # Adjust Z as needed

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

        if self._randomize_target_position:
            self.target_pos = self._generate_target_position()

        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation
