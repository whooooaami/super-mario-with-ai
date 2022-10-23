from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from IPython import embed
import gym
from gym.wrappers import *
import matplotlib.pyplot as plt

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayScaleObservation(env)

done = False
state = env.reset()

steps = 0
while not done:
    state, reward, done, info = env.step(env.action_space.sample())
    num_lives = info["life"]

    if num_lives < 2:
        done = True

    if done:
        break

    steps += 1
    state  = state / 255.
    env.render()

env.close()

print(steps)