#!/usr/bin/env python
#################################################################################
#Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import os
import json
import numpy as np
import random
import time
import sys

# (modified) import K
import keras.backend as K
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from src.turtlebot3_dqn.environment_stage_4 import Env
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop

# (modified) import Lambda layer
from keras.layers import Dense, Dropout, Activation, Lambda

EPISODES = 3000

from itertools import repeat as _repeat
import operator
from bisect import bisect as _bisect

def accumulate(iterable, func=operator.add):
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

def choices(population, weights=None, cum_weights=None, k=1):
    n = len(population)
    if cum_weights is None:
        if weights is None:
            _int = int
            n += 0.0
            return [population[_int(random.random() * n)] for i in _repeat(None, k)]
        cum_weights = list(accumulate(weights))
    elif weights is not None:
        raise TypeError('Cannot specify both weights and cumulative weights')
    if len(cum_weights) != n:
        raise TypeError('The number of weights does not match the population')
    bisect = _bisect
    total = cum_weights[-1] + 0.0
    hi = n - 1
    return [population[bisect(cum_weights, random.random() * total, 0, hi)] for i in _repeat(None, k)]

class ReinforceAgent():
    def __init__(self, state_size, action_size):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/stage_4_')
        self.result = Float32MultiArray()

        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.memory = deque(maxlen=1000000)
	#(edit5)
	self.priority = deque(maxlen=1000000)
	self.alpha = 0.7
	self.beta = 0.5
	#(edit5)

        self.model = self.buildModel()
        self.target_model = self.buildModel()

        self.updateTargetModel()

        if self.load_model:
            self.model.set_weights(load_model(self.dirPath + str(self.load_episode) + ".h5").get_weights())

            with open(self.dirPath + str(self.load_episode) + '.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

    def buildModel(self):
        model = Sequential()
        dropout = 0.2

        model.add(Dense(64, input_shape=(self.state_size,), activation='relu', kernel_initializer='he_uniform'))

        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(dropout))

        # (modified) self.action_size --> self.action_size + 1 to incorporate V + A
        model.add(Dense(self.action_size + 1, kernel_initializer='he_uniform'))
        model.add(Activation('linear'))
        model.add(Lambda(lambda x: K.expand_dims(x[:, 0], -1) + x[:, 1:] - K.mean(x[:, 1:], axis=1, keepdims=True), output_shape=(self.action_size,)))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()

        return model

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward

        # bootstrapping with optimization
        else:
            return reward + self.discount_factor * np.amax(next_target)

    # to update model when wanted to
    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    # using e-greedy
    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            # predict from kera.Sequential model
            # put in state and retrieve q-values for all actions
            q_value = self.model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            return np.argmax(q_value[0])


    def appendMemory(self, state, action, reward, next_state, done):
	#(edit5)
	td_error = reward + self.discount_factor * np.max(self.target_model.predict(next_state.reshape(1, len(next_state)))) - self.model.predict(state.reshape(1, len(state)))[0][action]
	
	p = np.abs(td_error + 0.0000001) ** self.alpha
	#(edit5)

        self.memory.append((state, action, reward, next_state, done))

	#(edit5)
	self.priority.append(p)	
	#(edit5)
    
    #(edit5)
    def get_PER_batch(self):
	p_sum = np.sum(self.priority)
	prob = self.priority / p_sum
	
	sample_indices = choices(range(len(prob)), k=self.batch_size, weights=prob)

	importance = (1/prob) * (1/len(self.priority))
	importance = np.array(importance)[sample_indices]
	samples = np.array(self.memory)[sample_indices]
	
	return samples, importance	
	#(edit5)	

    def trainModel(self, target=False):
	#(edit5)
        #mini_batch = random.sample(self.memory, self.batch_size)
	mini_batch, importance = self.get_PER_batch()
	#(edit5)

        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

	#(edit5)
	importance_batch = np.empty((0,1), dtype = np.float64)
	#(edit5)


        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            next_states = mini_batch[i][3]
            dones = mini_batch[i][4]

	    #(edit5)
	    i = importance[i] ** self.beta

            q_value = self.model.predict(states.reshape(1, len(states)))
            self.q_value = q_value

            # use of target model
            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)))

            # and just the model
            else:
                next_target = self.model.predict(next_states.reshape(1, len(next_states)))

            # calc q-value with model on next_state
            next_q_value = self.getQvalue(rewards, next_target, dones)

            # append state as X_batch
            Y_sample = q_value.copy()

            # cannot understand Y_sample[0][actions]
            # oh
            # so if the actions weren't taken, the values for target will be just the original q-value
            # or if the actions have been taken, they will be replaced for the actions taken by [0][actions]
            # actions --> action seems more suitable

            Y_sample[0][actions] = next_q_value

	    #(edit5)
	    importance_batch = np.append(importance_batch, i)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)

	    #(edit6)
	    else:
                X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

	#(edit5) added sample_weight=importance_batch
        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0, sample_weight=importance_batch)

if __name__ == '__main__':
    rospy.init_node('turtlebot3_dqn_stage_4')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    # state_size = [scan_range ...., heading, current_distance, obstacle_min_range, obstacle_angle]
    # where there are 24 samples for scan_range
    state_size = 28
    action_size = 5

    env = Env(action_size)

    agent = ReinforceAgent(state_size, action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()

    for e in range(agent.load_episode + 1, EPISODES):
        # initialization
        done = False
        state = env.reset()
        score = 0

        for t in range(agent.episode_step):
            # 1. choose action
            # 2. step
            # 3. append memory
            action = agent.getAction(state)

            next_state, reward, done = env.step(action)

            agent.appendMemory(state, action, reward, next_state, done)

            # start updating using target model when enough memory is appended
            # training is only applied to current model : updating target model will
            # be shown later

            # note that target model is only started to be in use after target_model has been updated (2000)
            # note that training model only starts after agent memory is large enough (64)
            if len(agent.memory) >= agent.train_start:
                if global_step <= agent.target_update:
                    agent.trainModel()
                else:
                    # True : don't use target model yet
                    agent.trainModel(True)

            score += reward
            state = next_state
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

            # saving model for later ref
            if e % 10 == 0:
                agent.model.save(agent.dirPath + str(e) + '.h5')
                with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                    json.dump(param_dictionary, outfile)

            # if the episode takes too much timesteps
            if t >= 500:
                rospy.loginfo("Time out!!")
                done = True

            if done:
                result.data = [score, np.max(agent.q_value)]
                pub_result.publish(result)
                agent.updateTargetModel()   # (fix) move line to below
                                            # (fix) or just update for after every episode?
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              e, score, len(agent.memory), agent.epsilon, h, m, s)
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                break

            global_step += 1
            # agent.target_update = 2000 : maybe too big?
            if global_step % agent.target_update == 0:
                rospy.loginfo("UPDATE TARGET NETWORK")
		agent.updateTargetModel()
                # (fix) agent.updateTargetModel()
                # (fix) and this is just for every timestep?

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
