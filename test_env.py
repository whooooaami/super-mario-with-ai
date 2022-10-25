from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from IPython import embed
import gym
from gym.wrappers import *
import matplotlib.pyplot as plt
from ppo import Agent
import numpy as np
import torch

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayScaleObservation(env)
env = gym.wrappers.FrameStack(env, 4)

done = False
state = env.reset()

agent = Agent().cuda()

agent.load_state_dict(torch.load("ppo-12648448.pt"))

agent.eval()

steps = 0
obs = env.reset()
while not done:
    action, logprob, _, value = agent.get_action_and_value(torch.Tensor(obs).unsqueeze(0).cuda())
    obs, reward, done, info = env.step(action.item())
    num_lives = info["life"]

    if num_lives < 2:
        done = True

    if done:
        break

    steps += 1
    env.render()

env.close()
print(steps)