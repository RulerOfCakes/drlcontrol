from envs.ant import CustomAntEnv
from envs.viperx import ViperXEnv
from envs.spot import SpotEnv
from envs.cassie import CassieEnv

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
    id="Spot-v0",
    entry_point="envs:SpotEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Cassie-v0",
    entry_point="envs:CassieEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
