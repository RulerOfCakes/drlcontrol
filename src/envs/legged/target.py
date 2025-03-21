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


class LeggedTargetEnv(LeggedEnv):
    """
    Can use any legged robot model, with the forward direction assumed to be +x.

    The goal of this environment is to achieve stable locomotion towards a given target position.
    A site by the name of "target_site" must be defined as a direct descendent of <worldbody/> in the given mjcf file.
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
        xml_file: str = "ant_target.xml",
        frame_skip: int = 5,  # each 'step' of the environment corresponds to 5 timesteps in the simulation
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 2.0,  # reward for getting closer to the target
        success_reward_weight: float = 1000.0,  # reward for reaching the target
        termination_cost: float = 1000.0,  # penalty for terminating the episode early
        ctrl_cost_weight: float = 0.001,  # penalize large/jerky actions
        main_body: Union[int, str] = 1,
        terminate_when_unhealthy: bool = True,  # terminate the episode when the robot is unhealthy
        termination_contacts: List[Union[int, str]] = [1],
        reset_noise_scale: float = 0.005,  # noise scale for resetting the robot's position
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = False,
        randomize_target_position: bool = True,
        initial_target_range: float = 5.0,
        max_target_range=10.0,  # absolute coordinate(x, y) range for the target position
        target_range_increment: float = 0.1,
        increment_frequency: int = 1000,  # Environment steps per increment (# of calls to `step()`)
        **kwargs,
    ):
        # initialize the reward variables
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._success_reward_weight = success_reward_weight
        self._termination_cost = termination_cost

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
        )

        self._max_target_range = max_target_range
        self._target_range = initial_target_range
        self._target_range_increment = target_range_increment
        self._increment_frequency = increment_frequency
        self._steps = 0

        self._randomize_target_position = randomize_target_position
        self.target_pos = self._generate_target_position()

        self.target_site_id = None  # site ID for the target position

        self.metadata = LeggedTargetEnv.metadata

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

        # load observation size from parent class
        obs_size = self.observation_space.shape[0]

        # relative target position vector (x,y)
        obs_size += 2

        # modify metadata for the final observation space
        self.observation_structure["relative_target_pos"] = 2

        # update the observation space
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self._prev_pos = (
            self.data.body(self.body_cfg.main_body).xpos[:2].copy()
        )  # previous position of the robot

    @property
    def goal_reached(self) -> bool:
        distance_to_target = np.linalg.norm(
            self.target_pos - self.data.body(self.body_cfg.main_body).xpos[:2]
        )
        return distance_to_target < 0.2

    ## Rewards

    def _reward_termination(self):
        return (
            self._terminate_when_unhealthy and (self.is_terminated)
        ) * self._termination_cost

    def _reward_success(self):
        return self.goal_reached * self._success_reward_weight

    def _reward_forward(self):
        current_pos = self.data.body(self.body_cfg.main_body).xpos[:2].copy()
        displacement = current_pos - self._prev_pos
        target_direction = self.target_pos - current_pos
        target_direction /= np.linalg.norm(target_direction)

        # reward for moving in the direction of the target
        displacement_on_target_direction = np.dot(displacement, target_direction)
        return self._forward_reward_weight * displacement_on_target_direction

    # z coordinate is omitted as the robot is expected to move in the x-y plane
    def _generate_target_position(self):
        x = self.np_random.uniform(low=-self._target_range, high=self._target_range)
        y = self.np_random.uniform(low=-self._target_range, high=self._target_range)
        return np.array([x, y], dtype=np.float32)

    def _get_obs(self):
        # get the current state of the robot
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        relative_target_pos = (
            self.target_pos - self.data.body(self.body_cfg.main_body).xpos[:2]
        )

        # exclude the x and y coordinates of the torso(root link) for position agnostic behavior in policies.
        if self.obs_cfg.exclude_current_positions_from_observation:
            qpos = qpos[2:]

        obs = np.concatenate([qpos, qvel, relative_target_pos])

        # include the external forces acting on the robot
        if self.obs_cfg.include_cfrc_ext_in_observation:
            obs = np.concatenate([obs, self.cfrc_ext])

        return obs

    def _get_rew(self, action: np.ndarray):
        forward_reward = self._reward_forward()
        success_reward = self._reward_success()
        rewards = forward_reward + success_reward

        ctrl_cost = self._reward_control(action)
        action_rate_cost = self._reward_action_rate(self._prev_action, action)
        termination_cost = self._reward_termination()
        costs = ctrl_cost + termination_cost + action_rate_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_success": success_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_termination": -termination_cost,
            "reward_action_rate": -action_rate_cost,
        }

        self._prev_pos = self.data.body(self.body_cfg.main_body).xpos[:2].copy()
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

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)

        # termination condition 1. unhealthy
        terminated = (self.is_terminated) and self._terminate_when_unhealthy
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
        if self._randomize_target_position:
            self.target_pos = self._generate_target_position()

        observation = super().reset_model()

        return observation
