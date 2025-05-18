import gym
import matplotlib.pyplot as plt
import numpy as np
np.bool8 = np.bool_
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env
# Actor-Critic Networks
dim_state = 3
dim_action = 1

class Actor(nn.Module):
    def __init__(self, dim_state=dim_state, dim_action=dim_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim_action),
        )
        #self.log_std = nn.Parameter(torch.tensor([-0.693], dtype=torch.float32))

    def forward(self, x):
        mu = self.net(x)
        # std = self.log_std.exp().expand_as(mu)
        return mu

class Critic(nn.Module):
    def __init__(self, dim_state=dim_state, dim_action=dim_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim_action),
        )

    def forward(self, x):
        return self.net(x)
    
class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.advantages = []
        self.rewards = []
    def add(self, state, action, log_prob, advantage, reward):

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        log_prob = torch.tensor(log_prob, dtype=torch.float32)
        advantage = torch.tensor(advantage, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.advantages.append(advantage)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.advantages = []
        self.rewards = []
    def get(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        log_probs = torch.stack(self.log_probs)
        advantages = torch.stack(self.advantages)
        rewards = torch.stack(self.rewards)
        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'advantages': advantages,
            'rewards': rewards
        }

# PPO Agent
class PPO:
    def __init__(self, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(5, 1).to(self.device)
        self.critic = Critic(5, 1).to(self.device)
        self.memory = ReplayBuffer()
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])
           
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.batch_size = 32
        self.lambda_ = 0.95
        self.std = torch.diag(torch.full(size=(1,), fill_value=0.5))

    def update(self):

        data = self.memory.get()
        states = data['states'].to(self.device)
        actions = data['actions'].to(self.device)
        log_probs = data['log_probs'].to(self.device)
        advantages = data['advantages'].to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        rewards = data['rewards'].to(self.device)

        dataset = TensorDataset(states, actions, log_probs, advantages, rewards)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(50):
            for state, action, old_log_prob, advantage, reward in dataloader:
                mu = self.actor(state)

                dist = Normal(mu, self.std)
                new_log_prob = dist.log_prob(action).sum(dim=-1)

                ratio = (new_log_prob - old_log_prob).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

                loss_actor = -torch.min(surr1, surr2).mean()
                # Ensure critic output and target have same shape [batch_size]
                value_pred = self.critic(state).squeeze(-1)
                loss_critic = nn.MSELoss()(value_pred, reward)
                 
                total_loss = loss_actor + 0.5 * loss_critic + 0.01 * dist.entropy().mean()
                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                self.optimizer.step()

        self.memory.clear()
    
    def train(self, env, num_episodes=10, epochs=100):

        self.actor.train()
        self.critic.train()
       
        total_rewards = []
        best_avg_reward = float('-inf')
        for epoch in range(epochs):
            print("epoch: ", epoch)
            for _ in range(num_episodes):

                states = []
                actions = []
                log_probs = []
                rewards = []

                state, _ = env.reset()
                total_reward = 0
                done = False

                while not done:
                    # Convert state to tensor so as to input into nn.Module
                    state = torch.tensor(state, dtype=torch.float32, device=self.device)
                    state = state.unsqueeze(0) # [1, dim_state]
                    mu = self.actor(state)
                    dist = Normal(mu, self.std)

                    # turn the action into numpy array, so as to input into env.step
                    action = dist.sample().squeeze(0) # [1, dim_action]
                    log_prob = dist.log_prob(action).sum().detach() # [1]
                    action = action.detach().cpu().numpy()

                    next_state, reward, done, truncated, _ = env.step(action)
                    done = done or truncated

                    states.append(state.squeeze(0))
                    actions.append(action)
                    log_probs.append(log_prob)
                    rewards.append(reward)

                    state = next_state
                    total_reward += reward
                    if done:
                        total_rewards.append(total_reward)
                        print("Reward: ", total_reward)

                values = self.critic(torch.stack(states)).detach() #[steps, 1]
                values = values.squeeze(1) # [steps]
                values = torch.cat([values, torch.zeros(1, device=self.device)]) # [steps + 1]

                GAE = torch.zeros(1, device=self.device)
                advantages = []
                for t in reversed(range(len(rewards))):
                    delta = rewards[t] + self.gamma * values[t+1] - values[t]
                    GAE = delta + self.gamma * self.lambda_ * GAE
                    advantages.insert(0, GAE)

                advantages = torch.cat(advantages)


                # 3. 計算 returns
                returns = advantages + values[:-1]    # [T]
                for s, a, lp, adv, ret in zip(states, actions, log_probs, advantages, returns):
                    self.memory.add(s, a, lp, adv, ret)

            self.update()
            if (epoch + 1) % 10 == 0:
                eval_rewards = []
                for _ in range(50):
                    state, _ = env.reset()
                    total_r = 0
                    done = False
                    while not done:
                        # 確保 state 是形狀 (1, 3)
                        state_array = np.array(state).reshape(1, -1)
                        state_tensor = torch.tensor(state_array, dtype=torch.float32, device=self.device)
                        mu = self.actor(state_tensor)
                        action = mu.detach().cpu().numpy()
                        state, r, done, truncated, _ = env.step(action)
                        done = done or truncated
                        total_r += r
                    eval_rewards.append(total_r)
                avg_reward = sum(eval_rewards) / len(eval_rewards)
                print(f"Evaluation at epoch {epoch+1}: avg_reward = {avg_reward}")
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save({
                        'actor_state_dict': self.actor.state_dict(),
                        'critic_state_dict': self.critic.state_dict()
                    }, "best_ppo_model.pth")
                    print(f"New best model saved with avg_reward = {avg_reward}")

        # Save the model
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, "last_ppo_model.pth")
        print("Model saved to ppo_model.pth")

        # Plot the rewards
        plt.plot(total_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('PPO Training Rewards')
        plt.savefig('ppo_training_rewards.png')

if __name__ == "__main__":
    # Action space: Box(-1.0, 1.0, (1,), float64)
    # Observation space: Box(-inf, inf, (5,), float64)

    env_name = "cartpole-balance"
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    agent = PPO()
    agent.train(env, num_episodes=10, epochs=100)
    env.close()