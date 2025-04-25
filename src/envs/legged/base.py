from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
import numpy as np

from utils.linalg import rotation_matrix

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
    termination_height_range: tuple[float, float] = (-np.inf, np.inf)
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
    include_cvel_in_observation: bool = (
        False  # include center of mass velocity in the observation
    )
    include_cinert_in_observation: bool = False  # include inertia in the observation
    include_qfrc_actuator_in_observation: bool = (
        False  # include actuator forces in the observation
    )
    include_circular_terrain_profile: bool = False
    include_forward_terrain_profile: bool = False
    contact_force_range: tuple[float, float] = (
        -np.inf,
        np.inf,
    )  # range of contact forces to be clipped in the observation
    terrain_profile_circular_radius: float = (
        0.0  # radius of the terrain profile around the robot
    )
    terrain_profile_circular_resolution: int = (
        0  # resolution of the terrain profile around the robot
    )

    terrain_profile_ray_origin: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )  # origin of the ray for terrain profile
    terrain_profile_ray_direction: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )  # direction of the ray for terrain profile
    terrain_profile_ray_length: float = (
        0.0  # length between the origin and the terrain profile window
    )
    terrain_profile_ray_dimension: tuple[float, float] = (
        0.0,
        0.0,
    )  # dimension of the ray for terrain profile
    terrain_profile_ray_resolution: tuple[int, int] = (1, 1)


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
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if isinstance(name, str)
                else name
            )
            for name in body_cfg.termination_contacts
        ]
        self._penalized_contact_indices = [
            (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
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
        obs_size += len(self.cvel) * self.obs_cfg.include_cvel_in_observation
        obs_size += (
            len(self.actuator_forces)
            * self.obs_cfg.include_qfrc_actuator_in_observation
        )
        obs_size += len(self.cinert) * self.obs_cfg.include_cinert_in_observation
        obs_size += (
            self.obs_cfg.terrain_profile_circular_resolution**2
        ) * self.obs_cfg.include_circular_terrain_profile
        obs_size += (
            self.obs_cfg.terrain_profile_ray_resolution[0]
            * self.obs_cfg.terrain_profile_ray_resolution[1]
        ) * self.obs_cfg.include_forward_terrain_profile

        # metadata for the final observation space
        # both observation_structure and observation_space must be correctly adjusted by child classes
        self.observation_structure = {
            "qpos": self.data.qpos.size
            - 2 * self.obs_cfg.exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cfrc_ext": len(self.data.cfrc_ext.ravel())
            * self.obs_cfg.include_cfrc_ext_in_observation,
            "cvel": len(self.cvel) * self.obs_cfg.include_cvel_in_observation,
            "actuator_forces": len(self.actuator_forces)
            * self.obs_cfg.include_qfrc_actuator_in_observation,
            "cinert": len(self.cinert) * self.obs_cfg.include_cinert_in_observation,
            "circular_terrain_profile": (
                self.obs_cfg.terrain_profile_circular_resolution**2
            )
            * self.obs_cfg.include_circular_terrain_profile,
            "forward_terrain_profile": (
                self.obs_cfg.terrain_profile_ray_resolution[0]
                * self.obs_cfg.terrain_profile_ray_resolution[1]
            )
            * self.obs_cfg.include_forward_terrain_profile,
        }

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self._prev_action = np.zeros(self.action_space.shape[0], dtype=np.float64)

    ### Observations
    @property
    def cvel(self):
        """
        Returns the center-of-mass based linear and angular velocities of the robot.
        """
        return self.data.cvel.flat.copy()

    @property
    def cinert(self):
        """
        Returns the inertia of the robot.
        """
        return self.data.cinert.flat.copy()

    @property
    def actuator_forces(self):
        """
        Returns a 1D array of actuator forces acting on the robot.
        """
        return self.data.qfrc_actuator.flat.copy()

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
        Check if the episode is terminated based on termination conditions.
        """
        # Check for contacts with the ground
        robot_height = self.data.body(self.body_cfg.main_body).xpos[2]
        return np.any(
            np.abs(self.contact_forces[self._termination_contact_indices].flatten()) > 0
        ) or (
            robot_height < self.body_cfg.termination_height_range[0]
            or robot_height > self.body_cfg.termination_height_range[1]
        )

    def terrain_profile_circular(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the terrain profile in a circular pattern around the robot.
        """

        radius = self.obs_cfg.terrain_profile_circular_radius
        resolution = self.obs_cfg.terrain_profile_circular_resolution

        if resolution <= 0 or radius <= 0:
            raise ValueError(
                "Resolution and radius must be greater than 0 for circular terrain profile."
            )

        robot_pos = self.data.body(self.body_cfg.main_body).xpos[:3]
        robot_x, robot_y, robot_z = robot_pos[0], robot_pos[1], robot_pos[2]
        profile_coords = [
            (robot_x + x, robot_y + y)
            for x in np.linspace(-radius, radius, resolution)
            for y in np.linspace(-radius, radius, resolution)
        ]

        terrain_profiles = []
        for x, y in profile_coords:
            # Cast a ray from the robot to the terrain profile point
            ray_origin = np.array([x, y, robot_z]).reshape(3, 1)
            ray_direction = np.array([0.0, 0.0, -1.0]).reshape(3, 1)
            intersection_geoms = np.zeros((1, 1), dtype=np.int32)

            result = mujoco.mj_ray(
                self.model,
                self.data,
                ray_origin,
                ray_direction,
                None,
                1,
                self.body_cfg.main_body,
                intersection_geoms,
            )

            if result != -1:
                terrain_profiles.append(result)
            else:
                terrain_profiles.append(0.0)  # arbitrary default value

        # modify profile coords to 3d ndarray format
        formatted_profile_coords = []
        for i in range(0, len(profile_coords)):
            x, y = profile_coords[i]
            z_dist = terrain_profiles[i]
            formatted_profile_coords.append(np.array([x, y, robot_z - z_dist]))

        return (
            np.array(formatted_profile_coords),
            np.array(terrain_profiles, dtype=np.float64),
        )

    def terrain_profile_ray(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the raycast terrain profile(depthmap) from the robot.
        """
        origin = self.obs_cfg.terrain_profile_ray_origin
        direction = self.obs_cfg.terrain_profile_ray_direction
        length = self.obs_cfg.terrain_profile_ray_length
        dimension = self.obs_cfg.terrain_profile_ray_dimension
        resolution = self.obs_cfg.terrain_profile_ray_resolution

        # convert local frame coordinate to world frame coordinate
        body_rot = self.data.body(self.body_cfg.main_body).xmat.copy().reshape(3, 3)
        origin = body_rot @ origin + self.data.body(self.body_cfg.main_body).xpos[:3]
        direction = body_rot @ direction
        direction = direction / np.linalg.norm(direction)

        window_center = origin + direction * length
        dx = np.cross(direction, np.array([0.0, 0.0, 1.0]))
        dx = dx / np.linalg.norm(dx) * dimension[0] / 2.0
        dy = np.cross(direction, dx)
        dy = dy / np.linalg.norm(dy) * dimension[1] / 2.0
        x_coords = np.linspace(-dx, dx, resolution[0])
        y_coords = np.linspace(-dy, dy, resolution[1])
        terrain_profile_coords = [
            (window_center + x + y) for x in x_coords for y in y_coords
        ]
        terrain_profiles = []

        for v in terrain_profile_coords:
            # Cast a ray from the robot to the terrain profile point
            ray_origin = origin
            ray_direction = v - ray_origin
            ray_length = np.linalg.norm(ray_direction)

            intersection_geoms = np.zeros((1, 1), dtype=np.int32)

            result = mujoco.mj_ray(
                self.model,
                self.data,
                ray_origin,
                ray_direction,
                None,
                1,
                self.body_cfg.main_body,
                intersection_geoms,
            )

            if result != -1 and result < ray_length:
                terrain_profiles.append(result)
            else:
                terrain_profiles.append(0.0)

        return (
            np.array(terrain_profile_coords),
            np.array(terrain_profiles, dtype=np.float64),
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

    def _reward_lin_vel_z(self, prev_pos: np.ndarray, pos: np.ndarray) -> float:
        """
        Penalize the robot for z axis linear velocity.
        """
        z_vel: float = (pos[2] - prev_pos[2]) / self.dt
        return np.square(z_vel)

    def _get_ground_height(self, x: float, y: float) -> float:
        INF_HEIGHT = 100.0
        ray_start = np.array([x, y, INF_HEIGHT], dtype=np.float64).reshape(3, 1)
        ray_dir = np.array([0, 0, -1], dtype=np.float64).reshape(3, 1)
        intersection_geoms = np.zeros((1, 1), dtype=np.int32)
        result = mujoco.mj_ray(
            self.model,
            self.data,
            ray_start,
            ray_dir,
            None,
            1,  # include static geoms
            self.body_cfg.main_body,
            intersection_geoms,
        )
        if result != -1:
            intersection_distance_from_body_height = result
            return INF_HEIGHT - intersection_distance_from_body_height
        else:
            return -INF_HEIGHT

    def _render_terrain_profile_circular(
        self,
        profile_coords: np.ndarray,
        color: np.ndarray = np.array([0.0, 0.0, 1.0, 1.0]),
    ):
        if self.render_mode == "human":
            for pos in profile_coords:
                self._render_site(pos=pos, color=color)

    def _render_terrain_profile_ray(
        self,
        ray_targets: np.ndarray,
        color: np.ndarray = np.array([0.0, 0.0, 1.0, 1.0]),
    ):
        # Render the terrain profile ray
        if self.render_mode == "human":
            for ray_target in ray_targets:
                self._render_arrow(
                    origin=self.obs_cfg.terrain_profile_ray_origin,
                    dir=ray_target,
                    length=np.linalg.norm(
                        ray_target
                    ),  # TODO: fix this with actual raycast result
                    color=color,
                )

    # INFO: A manual patch for self.mujoco_renderer.viewer.add_marker is required until the following PR is released:
    # https://github.com/Farama-Foundation/Gymnasium/pull/1329
    def _render_site(
        self,
        pos: np.ndarray,
        color: np.ndarray = np.array([0.0, 1.0, 0.0, 1.0]),
        radius: float = 0.05,
    ):
        # Render the site at the given position
        if self.render_mode == "human":
            self.mujoco_renderer.viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=pos,
                mat=np.eye(3).flatten(),
                size=np.array([radius, radius, radius]),
                rgba=color,
            )

    def _render_arrow(
        self,
        origin: np.ndarray,
        dir: np.ndarray,
        length: float = 1.0,
        color: np.ndarray = np.array([0.0, 1.0, 0.0, 1.0]),
    ):
        if self.render_mode != "human":
            return
        # Create a 3D arrow using the provided origin, direction, and length
        dir = (
            dir / np.linalg.norm(dir) if np.linalg.norm(dir) > 0 else np.zeros_like(dir)
        )

        up_dir = np.array([0, 0, 1])  # default orientation of mujoco arrow

        rotation = rotation_matrix(
            np.cross(up_dir, dir),
            np.arccos(np.dot(up_dir, dir)),
        ).flatten()

        # Add the arrow to the viewer
        self.mujoco_renderer.viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            pos=origin,
            mat=rotation,
            size=np.array([0.05, 0.05, length]),
            rgba=color,
        )

    def _get_obs(self):
        # get the current state of the robot
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # exclude the x and y coordinates of the torso(root link) for position agnostic behavior in policies.
        if self.obs_cfg.exclude_current_positions_from_observation:
            qpos = qpos[2:]

        obs = np.concatenate([qpos, qvel])

        if self.obs_cfg.include_cfrc_ext_in_observation:
            obs = np.concatenate([obs, self.data.cfrc_ext])

        if self.obs_cfg.include_cinert_in_observation:
            obs = np.concatenate([obs, self.cinert])

        if self.obs_cfg.include_qfrc_actuator_in_observation:
            obs = np.concatenate([obs, self.actuator_forces])

        if self.obs_cfg.include_cvel_in_observation:
            obs = np.concatenate([obs, self.cvel])

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
