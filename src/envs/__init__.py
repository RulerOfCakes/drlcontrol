from envs.ant import CustomAntEnv
from envs.viperx import ViperXEnv
from envs.spot import *
from envs.cassie import CassieEnv
from envs.go2 import *

from gymnasium.envs.registration import register

register(
    id="CustomAntEnv-v0",
    entry_point="envs:CustomAntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="ViperX-v0",
    entry_point="envs:ViperXEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Go-v0",
    entry_point="envs:GoEnvV0",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Spot-v0",
    entry_point="envs:SpotEnvV0",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Spot-v1",
    entry_point="envs:SpotEnvV1",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Spot-v2",
    entry_point="envs:SpotEnvV2",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Cassie-v0",
    entry_point="envs:CassieEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
