import gym
import gym_gridworlds
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr


env = gym.make('Cliff-v0')

# Q-table
Q = np.zeros([env.height , env.width, env.action_space.n])

# define hyperparameters (epsilon defined as function later)
dis = .90
num_episodes = 2000
lr = 0.01
success = 0

# create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    (y, x) = env.reset()
    # state = y , x
    rAll = 0
    done = False

    e = 1. / ((i // 100) + 1)  # Python2&3
    
    # The Q-Table learning algorithm
    while not done:

        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[y, x, :])
        # Get new state and reward from environment
        (new_y, new_x), reward, done, _ = env.step(action)
        # new_state = new_x , new_y

        # Update Q-Table with new knowledge using learning rate
        Q[y, x, action] = (Q[y, x, action] + lr * (reward + dis * np.max(Q[new_y, new_x, :]) - Q[y, x, action]))

        rAll += reward
        y, x = new_y, new_x

    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
