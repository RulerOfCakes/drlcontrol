from envs.ant import CustomAntEnv
from envs.viperx import ViperXEnv

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
