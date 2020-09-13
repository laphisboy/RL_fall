import gym
import numpy as np
import random

env = gym.make('CartPole-v1')
goal_steps = 500

while True:
    obs =  env.reset()
    for i in range(goal_steps):
        obs, reward, done, info = env.step(random.randrange(0,2))
        if done: break
        env.render()
        
