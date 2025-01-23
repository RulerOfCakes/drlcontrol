import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import gymnasium as gym
from torch.distributions import MultivariateNormal
import time
import math
import envs


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)

    return layer


class ActorNN(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(ActorNN, self).__init__()

        self.device = device

        self.fc = nn.Sequential(
            layer_init(nn.Linear(in_dim, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )

        self.mu = layer_init(nn.Linear(256, out_dim), std=0.01)
        self.sigma = nn.Linear(256, out_dim)

    def forward(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.Tensor(obs)
        if obs.device != self.device:
            obs = obs.to(self.device).float()
        x = self.fc(obs)
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))

        return mu, sigma


class CriticNN(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(CriticNN, self).__init__()

        self.device = device

        self.fc = nn.Sequential(
            layer_init(nn.Linear(in_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, out_dim), std=1),
        )

    def forward(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        elif obs.device != self.device:
            obs = obs.to(self.device).float()

        return self.fc(obs)


class PPO:
    def __init__(
        self,
        obs_dim,
        act_dim,
        timesteps_per_batch=10000,
        max_timesteps_per_episode=1000,
        ent_coef=0.0001,
        reward_scale=1.0,
        actor_lr=0.003,
        critic_lr=0.003,
        reward_normalize=False,
        device=None,
    ):
        # self.device = device if device is not None \
        #     else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device("cpu")

        print("device:", self.device)

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.actor = ActorNN(self.obs_dim, self.act_dim, self.device).to(self.device)
        self.critic = CriticNN(self.obs_dim, 1, self.device).to(self.device)

        print("actor params:", sum(p.numel() for p in self.actor.parameters()))
        print("critic params:", sum(p.numel() for p in self.critic.parameters()))

        self.timesteps_per_batch = timesteps_per_batch  # timesteps per batch
        self.max_timesteps_per_episode = (
            max_timesteps_per_episode  # timesteps per episode
        )

        # self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # self.cov_mat = torch.diag(self.cov_var)

        self.batch_obs = []  # batch observations
        self.batch_acts = []  # batch actions
        self.batch_log_probs = []  # log probs of each action
        self.batch_rewards = []  # batch rewards
        self.batch_vals = []  # batch value estimates
        self.batch_dones = []

        self.gamma = 0.99
        self.lam = 0.96

        self.clip_param = 0.2
        self.updates_per_iteration = 5
        self.ent_coef = ent_coef

        self.reward_scale = reward_scale
        self.reward_normalize = reward_normalize

        self.max_grad_norm = 0.5

        print("entropy coef:", self.ent_coef)

        self.num_mini_batches = 4

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, eps=1e-5
        )

    def insert(self, obs, act, log_prob, reward, val, done):
        self.batch_obs.append(obs)
        self.batch_acts.append(act)
        self.batch_log_probs.append(log_prob)
        self.batch_rewards.append(reward * self.reward_scale)  # scale rewards
        self.batch_vals.append(val)
        self.batch_dones.append(done)

    def get_action(self, obs: list[float]):
        obs = obs.copy()
        # obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        obs = torch.Tensor(obs)

        mean, cov = self.actor(obs)
        dist = MultivariateNormal(mean, torch.diag(cov))
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().cpu().numpy(), log_prob.detach()

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t + 1 == len(rewards):
                delta = rewards[t] - values[t]
            else:
                delta = (
                    rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                )

            advantage = delta + self.gamma * self.lam * last_advantage * (1 - dones[t])
            batch_advantages.insert(0, advantage)
            last_advantage = advantage

        batch_advantages = torch.tensor(
            batch_advantages, dtype=torch.float, device=self.device
        )
        returns = (
            batch_advantages
            + torch.tensor(values, dtype=torch.float, device=self.device).detach()
        )

        return returns, batch_advantages

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs)

        mean, cov = self.actor(batch_obs)
        dist = MultivariateNormal(mean, torch.diag_embed(cov))
        log_probs = dist.log_prob(batch_acts)

        return (
            V.squeeze(),
            log_probs,
            dist.entropy(),
            cov[torch.randint(0, batch_obs.shape[0], (1,))],
        )

    def learn(self):
        self.batch_obs = torch.tensor(
            self.batch_obs, dtype=torch.float, device=self.device
        )
        self.batch_acts = torch.tensor(
            self.batch_acts, dtype=torch.float, device=self.device
        )
        self.batch_log_probs = torch.tensor(
            self.batch_log_probs, dtype=torch.float, device=self.device
        ).flatten()
        self.batch_rewards = torch.tensor(
            self.batch_rewards, dtype=torch.float, device=self.device
        )

        with torch.no_grad():
            episodes = np.sum(self.batch_dones)

            print()
            print(
                "average episodic reward: ",
                (torch.sum(self.batch_rewards).item() / self.reward_scale)
                / max(1, episodes),
            )
            print(
                "average episodic length:", self.batch_obs.shape[0] / max(1, episodes)
            )

        if self.reward_normalize:
            self.batch_rewards = (self.batch_rewards - self.batch_rewards.mean()) / (
                self.batch_rewards.std() + 1e-10
            )

        returns, advantages = self.calculate_gae(
            self.batch_rewards, self.batch_vals, self.batch_dones
        )

        V = self.critic(self.batch_obs).squeeze()
        y_pred, y_true = V.detach().cpu().numpy(), returns.detach().cpu().numpy()
        explained_var = (
            np.nan
            if np.var(y_true) == 0
            else 1 - np.var(y_true - y_pred) / np.var(y_true)
        )

        print("variance of returns:", np.var(y_true))
        print("explained variance:", explained_var)
        print("difference in returns and pred_returns:", np.mean(y_true - y_pred))

        actor_losses = []
        critic_losses = []
        entropy_losses = []
        kl_approximations = []
        ratios_list = []

        train_start_time = time.time()

        steps = len(self.batch_obs)
        inds = np.arange(steps)
        mini_batch_size = math.ceil(steps / self.num_mini_batches)

        for _ in range(self.updates_per_iteration):
            # shuffle indices for mini-batch sampling
            np.random.shuffle(inds)

            for start in range(0, steps, mini_batch_size):
                end = min(start + mini_batch_size, steps)
                idx = inds[start:end]

                mini_obs = self.batch_obs[idx]
                mini_actions = self.batch_acts[idx]
                mini_log_probs = self.batch_log_probs[idx]
                mini_advantages = advantages[idx]
                mini_returns = returns[idx]

                # Normalize advantages by mini-batch
                mini_advantages = (mini_advantages - mini_advantages.mean()) / (
                    mini_advantages.std() + 1e-10
                )

                V, curr_log_probs, entropy, sample_cov = self.evaluate(
                    mini_obs, mini_actions
                )

                logratios = curr_log_probs - mini_log_probs
                ratios = torch.exp(logratios)

                ratios_list.append(ratios.mean().item())

                with torch.no_grad():
                    approx_kl = ((ratios - 1) - logratios).mean()
                    kl_approximations.append(approx_kl.item())

                surr1 = ratios * mini_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param)
                    * mini_advantages
                )

                # print('surr1 mean: ', surr1.mean())
                # print('surr2 mean: ', surr2.mean())

                actor_loss = (-torch.min(surr1, surr2)).mean()

                entropy_loss = entropy.mean()
                entropy_losses.append(entropy_loss.item())

                actor_loss = actor_loss - self.ent_coef * entropy_loss
                actor_losses.append(actor_loss.item())

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # print(V.shape)
                # print(batch_rtgs.shape)

                critic_loss = nn.MSELoss()(V, mini_returns)
                critic_losses.append(critic_loss.item())

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

            if _ == self.updates_per_iteration - 1:
                print("actor loss:", np.mean(actor_losses))
                print("critic loss:", np.mean(critic_losses))
                print("entropy loss:", np.mean(entropy_losses))
                print("entropy loss * coef:", np.mean(entropy_losses) * self.ent_coef)
                print("returns mean: ", torch.mean(returns).item())
                print("sample cov:", np.round(sample_cov.tolist(), 3))
                print("kl approx:", np.mean(kl_approximations))
                print("ratios:", np.mean(ratios_list))

        print("time taken to train:", time.time() - train_start_time)

        self.batch_obs = []
        self.batch_acts = []
        self.batch_log_probs = []
        self.batch_rewards = []
        self.batch_vals = []
        self.batch_dones = []

    def save(self):
        torch.save(self.actor.state_dict(), "actor.pth")
        torch.save(self.critic.state_dict(), "critic.pth")

        torch.save(self.actor_optimizer.state_dict(), "actor_optimizer.pth")
        torch.save(self.critic_optimizer.state_dict(), "critic_optimizer.pth")

    def load(self):
        self.actor.load_state_dict(torch.load("actor.pth"))
        self.critic.load_state_dict(torch.load("critic.pth"))

        self.actor_optimizer.load_state_dict(torch.load("actor_optimizer.pth"))
        self.critic_optimizer.load_state_dict(torch.load("critic_optimizer.pth"))


# create gym env

current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
models_path = os.path.join(parent_path, "models")
env = gym.make(
    "ViperX-v0",
    render_mode="human",
    frame_skip=5,
    max_episode_steps=500,  # physics steps will have been multiplied by 5, due to the frame_skip value
    xml_file=os.path.join(models_path, "trossen_vx300s/scene_box.xml"),
)

model = PPO(
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.shape[0],
    ent_coef=5e-5,
    device=torch.device("cpu"),
    actor_lr=0.0005,
    critic_lr=0.0005,
    timesteps_per_batch=4000,
    reward_scale=0.1,
)

total_t = 0

while True:
    obs, info = env.reset()
    done = False
    t = 0

    while True:
        t += 1
        action, log_prob = model.get_action(obs)

        val = model.critic(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated

        model.insert(obs, action, log_prob, reward, val.flatten(), done)

        if done:
            break

    total_t += t

    if total_t >= model.timesteps_per_batch:
        model.learn()
        total_t = 0
