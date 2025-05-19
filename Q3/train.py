import argparse
import datetime
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from collections import deque
import random
import os, sys
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env
# Utils

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, mask):
        # print(len(state), len(action), (reward), len(next_state), (mask))
        self.buffer.append((state, action, reward, next_state, mask))
    def sample(self, batch_size):
        states, actions, rewards, next_states, masks = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(masks)
    def __len__(self):
        return len(self.buffer)

# Networks
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q = self.net(x)
        return q

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
     
    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(-20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) \
                - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, x_t

# SAC Agent
class SAC:
    def __init__(self, state_dim, action_space, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        # Critics

        self.critic1 = QNetwork(state_dim, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic2 = QNetwork(state_dim, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic1_opt = Adam(self.critic1.parameters(), lr=args.lr)
        self.critic2_opt = Adam(self.critic2.parameters(), lr=args.lr)

        # Target critics
        self.critic1_target = QNetwork(state_dim, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic2_target = QNetwork(state_dim, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Policy
        self.policy = GaussianPolicy(state_dim, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_opt = Adam(self.policy.parameters(), lr=args.lr)

        # Entropy tuning
        if args.automatic_entropy_tuning:
            self.target_entropy = -np.prod(action_space.shape)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = Adam([self.log_alpha], lr=args.lr)
        else:
            self.alpha = args.alpha

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            mean, _ = self.policy(state)
            action = torch.tanh(mean)
            return action.detach().cpu().numpy()[0]
        else:
            action, _, _ = self.policy.sample(state)
            return action.detach().cpu().numpy()[0]

    def update(self, memory, batch_size, updates):
        states, actions, rewards, next_states, masks = memory.sample(batch_size)
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.FloatTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        mask_batch = torch.FloatTensor(masks).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state_batch)
            q1_next = self.critic1_target(next_state_batch, next_action)
            q2_next = self.critic2_target(next_state_batch, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            next_q_value = reward_batch + mask_batch * self.gamma * min_q_next

        q1 = self.critic1(state_batch, action_batch)
        q2 = self.critic2(state_batch, action_batch)
        critic1_loss = F.mse_loss(q1, next_q_value)
        critic2_loss = F.mse_loss(q2, next_q_value)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # Policy update
        action_new, log_prob, _ = self.policy.sample(state_batch)
        q1_new = self.critic1(state_batch, action_new)
        q2_new = self.critic2(state_batch, action_new)
        min_q_new = torch.min(q1_new, q2_new)
        policy_loss = ((self.alpha * log_prob) - min_q_new).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # Entropy temperature update
        if hasattr(self, 'log_alpha'):
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.)

        # Soft update
        if updates % args.target_update_interval == 0:
            soft_update(self.critic1_target, self.critic1, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)

        return critic1_loss.item(), critic2_loss.item(), policy_loss.item(), alpha_loss.item(), self.alpha.item()

# Training Loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="humanoid-walk")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--alpha", type=float, default=0.2) # Temperature parameter
    parser.add_argument("--automatic_entropy_tuning", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--num_steps", type=int, default=1000000)
    parser.add_argument("--start_steps", type=int, default=30000) # warmup steps
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--target_update_interval", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    import os
    os.makedirs(args.save_dir, exist_ok=True)

    def evaluate(agent, env, episodes=10):
        agent.policy.eval()
        rewards = []
        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                ep_reward += reward
                state = next_state
            rewards.append(ep_reward)
        agent.policy.train()
        avg_reward = sum(rewards) / len(rewards)
        print(f"Evaluation over {episodes} episodes: {avg_reward:.2f}")
        return avg_reward

    env = make_dmc_env(args.env, np.random.randint(0, 1000000), flatten=True, use_pixels=False)

    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    print("Env: ", env.observation_space.shape[0], env.action_space)
    memory = ReplayBuffer(args.buffer_size)

    total_steps = 0
    updates = 0
    best_reward = -float("inf")
    reward_list = []
    for episode in tqdm(range(1, 3001)):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            if total_steps < args.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            # print(f"Action: {action}")
            next_state, reward, done, truncated, _ = env.step(action)
            mask = 1 if not done else 0
            memory.push(state, action, reward, next_state, mask)
            done = done or truncated

            state = next_state
            episode_reward += reward
            total_steps += 1

            if len(memory) > args.start_steps:
                # print("update")
                losses = agent.update(memory, args.batch_size, updates)
                updates += 1

        # print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Steps: {total_steps}")
        reward_list.append(episode_reward)
        if episode % 10 == 0:
            print(f"Episode: {episode}, Average Reward: {np.mean(reward_list[-10:]):.2f}")
        if episode % 100 == 0:
            # Evaluate and save checkpoint
            avg_reward = evaluate(agent, env)
            if avg_reward > best_reward:
                best_reward = avg_reward
                print(f"New best reward: {best_reward:.2f}")
                torch.save(agent.policy.state_dict(), os.path.join(args.save_dir, f"best_policy_{episode}.pth"))
                print(f"Saved best policy to {args.save_dir}/best_policy_{episode}.pth")
