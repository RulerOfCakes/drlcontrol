import os
import gymnasium as gym

from stable_baselines3 import PPO
import envs

current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
models_path = os.path.join(parent_path, "models")

# Parallel environments
env = gym.make(
    "ViperX-v0",
    render_mode="human",
    frame_skip=5,
    max_episode_steps=500,  # physics steps will have been multiplied by 5, due to the frame_skip value
    xml_file=os.path.join(models_path, "trossen_vx300s/scene_box.xml"),
    collision_penalty_weight = 0.0,
    gripper_distance_reward_weight = 4.0,
    ctrl_cost_weight = 0.05,
    success_reward = 2000,
    box_distance_reward_weight = 6,
    grasp_reward_weight = 6,
    time_penalty_weight = 0
)

model = PPO(policy="MlpPolicy", env=env, verbose=1)
model.learn(total_timesteps=1e6)