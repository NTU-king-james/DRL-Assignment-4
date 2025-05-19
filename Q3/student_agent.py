import gymnasium as gym
import numpy as np
from train import GaussianPolicy
import torch
# Do not modify the input of the 'act' function and the '__init__' function. 

agent = GaussianPolicy(67, 21, hidden_size=512)
agent.load_state_dict(torch.load("best_policy_1100.pth", map_location='cpu'))
agent.eval()

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)

    def act(self, observation):
        agent_input = torch.from_numpy(observation).float() # zero copy, share memory
        mean, _ = agent(agent_input)
        mean = torch.tanh(mean)
        action = mean.detach().numpy()
        return action
