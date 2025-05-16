import gymnasium as gym
import numpy as np
import torch
from train import Actor, Critic, PPO
# Do not modify the input of the 'act' function and the '__init__' function. 

actor = Actor()
critic = Critic()
agent  = PPO(actor, critic)

checkpoint = torch.load("best_ppo_model.pth", map_location=agent.device)
actor.load_state_dict( checkpoint['actor_state_dict'] )
critic.load_state_dict( checkpoint['critic_state_dict'] )

actor.eval()
critic.eval()

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)

    def act(self, observation):
        mu = actor(torch.tensor(observation, dtype=torch.float32))
        return mu.detach().numpy()