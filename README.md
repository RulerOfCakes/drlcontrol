# DRLControl

Applying Deep Reinforcement Learning to control physically-based models.

This repository contains:
- Custom gymnasium environments
- Implementations of RL agents to solve these environments

## Environments

These custom environments are mostly built using [Gymnasium](https://gymnasium.farama.org/) and [Mujoco](https://mujoco.readthedocs.io/). They are typically more complicated than the ones provided by Gymnasium. They are designed to be more challenging and to require more complex control strategies. Feel free to use them to test your own implementations of RL agents.

Here are some of the custom environments included in this repository.

### ViperX - Robot Arm Manipulation

![ViperX 300 6DOF](assets/viperx.gif)

This environment aims to move the robot arm to grab the box and place it in the target location. It uses the ViperX 300 6DOF robot arm model from Trossen Robotics, provided from the [mujoco-menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trossen_vx300s) repository.