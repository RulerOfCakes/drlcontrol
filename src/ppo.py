"""
    The file contains the PPO class to train with.
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
            It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import time
import gymnasium as gym

import numpy as np
import math

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import envs, os


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)

    return layer


class ActorNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ActorNN, self).__init__()

        self.fc = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        self.mu = layer_init(nn.Linear(hidden_dim, out_dim), std=0.01)
        self.sigma = nn.Linear(hidden_dim, out_dim)

    def forward(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.Tensor(obs, dtype=torch.float32)
        x = self.fc(obs)
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))

        return mu, sigma


class CriticNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(CriticNN, self).__init__()

        self.fc = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, out_dim), std=1),
        )

    def forward(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32)

        return self.fc(obs)


class PPO:
    """
    This is the PPO class we will use as our model in main.py
    """

    def __init__(
        self,
        env,
        timestep_per_batch=4800,
        max_timesteps_per_episode=3200,
        n_updates_per_iteration=5,
        actor_hidden_dim=64,
        critic_hidden_dim=64,
        lr=2.5e-4,
        actor_lr_scale=0.5,
        gamma=0.99,
        clip=0.2,
        lam=0.95,
        num_minibatches=6,
        ent_coef=0,
        target_kl=0.02,
        max_grad_norm=0.5,
        save_freq=10,
        deterministic=False,
        seed=None,
        reward_scale=1.0,
    ):
        """
        Initializes the PPO model, including hyperparameters.

        Parameters:
            policy_class - the policy class to use for our actor/critic networks.
            env - the environment to train on.
            hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

        Returns:
            None
        """
        # Make sure the environment is compatible with our code
        assert type(env.observation_space) == gym.spaces.box.Box
        assert type(env.action_space) == gym.spaces.box.Box

        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = (
            timestep_per_batch  # Number of timesteps to run per batch
        )
        self.max_timesteps_per_episode = (
            max_timesteps_per_episode  # Max number of timesteps per episode
        )
        self.reward_scale = reward_scale  # Value to scale rewards by
        self.n_updates_per_iteration = n_updates_per_iteration  # Number of times to update actor/critic per iteration
        self.lr = lr  # Learning rate of actor optimizer
        self.gamma = (
            gamma  # Discount factor to be applied when calculating Rewards-To-Go
        )
        self.clip = clip  # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.lam = lam  # Lambda Parameter for GAE
        self.num_minibatches = (
            num_minibatches  # Number of mini-batches for Mini-batch Update
        )
        self.ent_coef = ent_coef  # Entropy coefficient for Entropy Regularization
        self.target_kl = target_kl  # KL Divergence threshold
        self.max_grad_norm = max_grad_norm  # Gradient Clipping threshold

        # Miscellaneous parameters
        self.save_freq = save_freq  # How often we save in number of iterations
        self.deterministic = deterministic  # If we're testing, don't sample actions
        self.seed = (
            seed  # Sets the seed of our program, used for reproducibility of results
        )

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert type(self.seed) == int

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

        # Extract environment information
        self.env = env

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Initialize actor and critic networks
        self.actor = ActorNN(self.obs_dim, actor_hidden_dim, self.act_dim)  # ALG STEP 1
        self.critic = CriticNN(self.obs_dim, critic_hidden_dim, 1)

        self.actor_lr_scale = actor_lr_scale

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(
            self.actor.parameters(), lr=self.lr * self.actor_lr_scale
        )
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        print("Actor Params:", sum(p.numel() for p in self.actor.parameters()))
        print("Critic Params:", sum(p.numel() for p in self.critic.parameters()))

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "i_so_far": 0,  # iterations so far
            "batch_lens": [],  # episodic lengths in batch
            "batch_rews": [],  # episodic returns in batch
            "explained_variance": 0,  # explained variance of value function
            "avg_batch_return": 0,  # average returns from GAE
            "avg_batch_est": 0,  # average return estimate from value network
            "actor_losses": [],  # losses of actor network in current iteration
            "critic_losses": [],  # losses of critic network in current iteration
            "entropy_losses": [],  # losses of entropy regularization in current iteration
            "approx_kl": [],  # approximated KL-divergence in current iteration
            "lr": 0,
        }

    def learn(self, total_timesteps):
        """
        Train the actor and critic networks. Here is where the main PPO algorithm resides.

        Parameters:
            total_timesteps - the total number of timesteps to train for

        Return:
            None
        """
        print(
            f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ",
            end="",
        )
        print(
            f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps"
        )
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        while t_so_far < total_timesteps:  # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            (
                batch_obs,
                batch_acts,
                batch_log_probs,
                batch_rews,
                batch_lens,
                batch_vals,
            ) = self.rollout()  # ALG STEP 3

            # Calculate advantage using GAE
            batch_advantages = self.calculate_gae(batch_rews, batch_vals)
            V = self.critic(batch_obs).squeeze()
            batch_returns = batch_advantages + V.detach()

            self.logger["avg_batch_return"] = torch.mean(batch_returns.detach()).item()
            self.logger["avg_batch_est"] = torch.mean(V.detach()).item()

            y_pred, y_true = (
                V.detach().cpu().numpy(),
                batch_returns.detach().cpu().numpy(),
            )
            self.logger["explained_variance"] = (
                np.nan
                if np.var(y_true) == 0
                else 1 - np.var(y_true - y_pred) / np.var(y_true)
            )

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                batch_advantages.std() + 1e-10
            )

            # This is the loop where we update our network for some n epochs
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = math.ceil(step / self.num_minibatches)
            actor_loss_list = []
            critic_loss_list = []
            entropy_loss_list = []
            approx_kl_list = []

            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Learning Rate Annealing
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)

                # Make sure learning rate doesn't go below 0
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr * self.actor_lr_scale
                self.critic_optim.param_groups[0]["lr"] = new_lr
                # Log learning rate
                self.logger["lr"] = new_lr

                # Mini-batch Update
                np.random.shuffle(inds)  # Shuffling the index
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    # Extract data at the sampled indices
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = batch_advantages[idx]
                    mini_rtgs = batch_returns[idx]

                    # Calculate V_phi and pi_theta(a_t | s_t) and entropy
                    V, curr_log_probs, entropy, sample_cov = self.evaluate(
                        mini_obs, mini_acts
                    )

                    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    # NOTE: we just subtract the logs, which is the same as
                    # dividing the values and then canceling the log with e^log.
                    # For why we use log probabilities instead of actual probabilities,
                    # here's a great explanation:
                    # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                    # TL;DR makes gradient descent easier behind the scenes.
                    logratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios - 1) - logratios).mean()

                    # Calculate surrogate losses.
                    surr1 = ratios * mini_advantage
                    surr2 = (
                        torch.clamp(ratios, 1 - self.clip, 1 + self.clip)
                        * mini_advantage
                    )

                    # Calculate actor and critic losses.
                    # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                    # the performance function, but Adam minimizes the loss. So minimizing the negative
                    # performance function maximizes it.
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)

                    # Entropy Regularization
                    entropy_loss = self.ent_coef * entropy.mean()
                    # Discount entropy loss by given coefficient
                    actor_loss = actor_loss - entropy_loss

                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    # Gradient Clipping with given threshold
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm
                    )
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm
                    )
                    self.critic_optim.step()

                    actor_loss_list.append(actor_loss.detach())
                    critic_loss_list.append(critic_loss.detach())
                    entropy_loss_list.append(entropy_loss.detach())

                    approx_kl_list.append(approx_kl.detach())

                # Approximating KL Divergence
                if approx_kl > self.target_kl:
                    break  # if kl aboves threshold

            # Log actor loss
            avg_actor_loss = sum(actor_loss_list) / len(actor_loss_list)
            avg_critic_loss = sum(critic_loss_list) / len(critic_loss_list)
            avg_entropy_loss = sum(entropy_loss_list) / len(entropy_loss_list)
            avg_approx_kl = sum(approx_kl_list) / len(approx_kl_list)

            self.logger["actor_losses"].append(avg_actor_loss)
            self.logger["critic_losses"].append(avg_critic_loss)
            self.logger["entropy_losses"].append(avg_entropy_loss)
            self.logger["approx_kl"].append(avg_approx_kl)

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                self.save_model()

    def calculate_gae(self, rewards, values):
        batch_advantages = []  # List to store computed advantages for each timestep

        # Iterate over each episode's rewards, values, and done flags
        for ep_rews, ep_vals in zip(rewards, values):
            advantages = []  # List to store advantages for the current episode
            last_advantage = 0  # Initialize the last computed advantage

            # Calculate episode advantage in reverse order (from last timestep to first)
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    # Calculate the temporal difference (TD) error for the current timestep
                    delta = ep_rews[t] + self.gamma * ep_vals[t + 1] - ep_vals[t]
                else:
                    # Special case at the boundary (last timestep)
                    delta = ep_rews[t] - ep_vals[t]

                # Calculate Generalized Advantage Estimation (GAE) for the current timestep
                advantage = delta + self.gamma * self.lam * last_advantage
                last_advantage = (
                    advantage  # Update the last advantage for the next timestep
                )
                advantages.insert(
                    0, advantage
                )  # Insert advantage at the beginning of the list

            # Extend the batch_advantages list with advantages computed for the current episode
            batch_advantages.extend(advantages)

        # Convert the batch_advantages list to a PyTorch tensor of type float
        return torch.tensor(batch_advantages, dtype=torch.float)

    def rollout(self):

        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        ep_vals = []
        t = 0  # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = []  # rewards collected per episode
            ep_vals = []  # state values collected per episode
            # Reset the environment. Note that obs is short for observation.
            obs, _ = self.env.reset()
            # Initially, the game is not done
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1  # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs)

                # Calculate action and make a step in the env.
                # Note that rew is short for reward.
                action, log_prob = self.get_action(obs)
                val = self.critic(obs)

                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # Track recent reward, action, and action log probability
                ep_rews.append(rew * self.reward_scale)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths, rewards, state values, and done flags
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten()

        # Log the episodic returns and episodic lengths in this batch.
        self.logger["batch_rews"] = batch_rews
        self.logger["batch_lens"] = batch_lens

        # Here, we return the batch_rews instead of batch_rtgs for later calculation of GAE
        return (
            batch_obs,
            batch_acts,
            batch_log_probs,
            batch_rews,
            batch_lens,
            batch_vals,
        )

    def get_action(self, obs):
        """
        Queries an action from the actor network, should be called from rollout.

        Parameters:
            obs - the observation at the current timestep

        Return:
            action - the action to take, as a numpy array
            log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        obs = torch.tensor(obs, dtype=torch.float)
        mean, cov = self.actor(obs)

        dist = MultivariateNormal(mean, torch.diag_embed(cov))

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # If we're testing, just return the deterministic action. Sampling should only be for training
        # as our "exploration" factor.
        if self.deterministic:
            return mean.detach().numpy(), 1

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

        Parameters:
            batch_obs - the observations from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of observation)
            batch_acts - the actions from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of action)
            batch_rtgs - the rewards-to-go calculated in the most recently collected
                            batch as a tensor. Shape: (number of timesteps in batch)
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        # if batch_obs.size(0) == 1:
        #     V = self.critic(batch_obs)
        # else:
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean, cov = self.actor(batch_obs)

        dist = MultivariateNormal(mean, torch.diag_embed(cov))
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return (
            V,
            log_probs,
            dist.entropy(),
            cov[torch.randint(0, batch_obs.shape[0], (1,))],
        )

    def _log_summary(self):
        """
        Print to stdout what we've logged so far in the most recent batch.

        Parameters:
            None

        Return:
            None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger["t_so_far"]
        i_so_far = self.logger["i_so_far"]
        lr = self.logger["lr"]
        avg_ep_lens = np.mean(self.logger["batch_lens"])
        avg_ep_rews = (
            np.mean([np.sum(ep_rews) for ep_rews in self.logger["batch_rews"]])
            / self.reward_scale
        )
        avg_actor_loss = np.mean(
            [losses.float().mean() for losses in self.logger["actor_losses"]]
        )
        avg_critic_loss = np.mean(
            [losses.float().mean() for losses in self.logger["critic_losses"]]
        )
        avg_entropy_loss = np.mean(
            [losses.float().mean() for losses in self.logger["entropy_losses"]]
        )
        avg_approx_kl = np.mean(
            [divergence.float().mean() for divergence in self.logger["approx_kl"]]
        )

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = f"{avg_ep_lens:.2f}"
        avg_ep_rews = f"{avg_ep_rews:.2f}"
        avg_batch_return = f"{self.logger['avg_batch_return']:.3f}"
        est_batch_return = f"{self.logger['avg_batch_est']:.3f}"
        explained_variance = f"{self.logger['explained_variance']:.4f}"
        avg_actor_loss = f"{avg_actor_loss:.7f}"
        avg_critic_loss = f"{avg_critic_loss:.7f}"
        avg_entropy_loss = f"{avg_entropy_loss:.7f}"
        avg_approx_kl = f"{avg_approx_kl:.5f}"
        lr = f"{lr:.7f}"

        # Print logging statements
        print(flush=True)
        print(
            f"-------------------- Iteration #{i_so_far} --------------------",
            flush=True,
        )
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Batch Return (By GAE): {avg_batch_return}", flush=True)
        print(
            f"Estimated Batch Return (By Critic Network): {est_batch_return}",
            flush=True,
        )
        print(f"Explained Variance: {explained_variance}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
        print(f"Average Critic Loss: {avg_critic_loss}", flush=True)
        print(f"Average Entropy Loss: {avg_entropy_loss}", flush=True)
        print(f"KL Divergence Approximation: {avg_approx_kl}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger["batch_lens"] = []
        self.logger["batch_rews"] = []
        self.logger["actor_losses"] = []
        self.logger["critic_losses"] = []
        self.logger["entropy_losses"] = []
        self.logger["approx_kl"] = []


    def save_model(self):
        torch.save(self.actor.state_dict(), "./ppo_actor.pth")
        torch.save(self.critic.state_dict(), "./ppo_critic.pth")

    
    def load_model(self):
        self.actor.load_state_dict(torch.load("./ppo_actor.pth", weights_only=True))
        self.critic.load_state_dict(torch.load("./ppo_critic.pth", weights_only=True))


# env = gym.make("Pendulum-v1")
# env = gym.make('Ant-v5')
# env = gym.make("Hopper-v5")
# env = gym.make("LunarLander-v3", continuous=True)

current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
models_path = os.path.join(parent_path, "models")

# env = gym.make(
#     "ViperX-v0",
#     render_mode="human",
#     frame_skip=5,
#     max_episode_steps=1000,  # physics steps will have been multiplied by 5, due to the frame_skip value
#     xml_file=os.path.join(models_path, "trossen_vx300s/scene_box.xml"),
#     collision_penalty_weight = 0.0,
#     success_reward = 2000,
# )

env = gym.make(
    "Spot-v0",
    render_mode="human",
    frame_skip=5,
    max_episode_steps=1000,  # physics steps will have been multiplied by 5, due to the frame_skip value
    xml_file=os.path.join(models_path, "boston_dynamics_spot/scene.xml"),
)

myppo = PPO(
    env,
    reward_scale=0.008,
    lr=1.5e-4,
    ent_coef=3e-4,
    timestep_per_batch=8000,
    actor_hidden_dim=512,
    critic_hidden_dim=512,
    n_updates_per_iteration=10,
    save_freq=10,
)

#myppo.load_model()
myppo.learn(100000000)
