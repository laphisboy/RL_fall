import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

# Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

# define hyperparameters (epsilon defined as function later)
dis = .90
num_episodes = 2000
lr = 0.01

# create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    e = 1. / ((i // 100) + 1)  

    if np.random.rand(1) < e:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    
    # The Q-Table learning algorithm
    while not done:
        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Choose the next action by e greedy
        if np.random.rand(1) < e:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(Q[new_state, :])

        # Update Q-Table with new knowledge using learning rate
        Q[state, action] = (Q[state, action] + lr * (reward + dis * Q[new_state, next_action] - Q[state, action]))

        rAll += reward
        state = new_state
        action = next_action

    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
