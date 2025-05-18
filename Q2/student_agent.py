import gymnasium
import numpy as np
import torch
from train import Actor, Critic, PPO

actor = Actor(5, 1)
critic = Critic(5, 1)
agent  = PPO(actor, critic)

checkpoint = torch.load("best_ppo_model.pth", map_location=agent.device)
actor.load_state_dict( checkpoint['actor_state_dict'] )
critic.load_state_dict( checkpoint['critic_state_dict'] )

actor.eval()
critic.eval()
# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)

    def act(self, observation):
        mu = actor(torch.tensor(observation, dtype=torch.float32))
        return mu.detach().numpy()

