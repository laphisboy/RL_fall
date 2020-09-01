# note that given implementation does not ensure success
# it should be inferred from result that if reward is not too small (negatively large)
# the agent has reached the end

# but it is enough to see the difference with Q-learning
# since it can show how often the agent falls down the cliff (negative reward in greater magnitude)


import gym
import gym_gridworlds
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr


env = gym.make('Cliff-v0')

# Q-table
# note that:
# action = [up, right, down, left]
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

    e = 1. / ((i // 100) + 1)
    # e-greedy where epsilon starts big
    # and becomes small
    # = more exploration in the beginning and less later on
    # with larger num_episodes, this might come into better effect

    # note that this is a method to satisfy condition of GLIE

    # e-greedy
    if np.random.rand(1) < e:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[y, x, :])
    
    # The Q-Table learning algorithm
    while not done:
        # Get new state and reward from environment
        (new_y, new_x), reward, done, _ = env.step(action)
        # new_state = new_x , new_y
        # Choose the next action by e greedy
        if np.random.rand(1) < e:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(Q[new_y, new_x, :])

        # Update Q-Table with new knowledge using learning rate
        Q[y, x, action] = (Q[y, x, action] + lr * (reward + dis * Q[new_y, new_x, next_action] - Q[y, x, action]))

        rAll += reward
        y, x = new_y, new_x

        # difference with Q-learning
        # action is already chosen for updating current state-action value
        action = next_action
        # therefore the loop continues by choosing only the next action

    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
