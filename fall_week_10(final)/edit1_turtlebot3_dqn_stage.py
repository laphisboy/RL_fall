#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
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
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from src.turtlebot3_dqn.environment_stage_3 import Env
from keras.models import Model, Sequential, load_model, clone_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Softmax, Activation, Input
# might have to change above keras to tensorflow.keras
# or 
# import tensorflow.keras as keras
# from keras ... 

#(edit1)
import tensorflow as tf

EPISODES = 3000


# (edit1) from OpenAI baseline a2c.utils
# not verified if it works yet...
def get_by_index(x, idx):
    assert(len(x.get_shape())==2)
    print(idx)
    print(idx.get_shape())
    assert(len(idx.get_shape())==1)
    idx_flattened = tf.range(0, x.shape[0]) * x.shape[1] + idx
    y = tf.gather(tf.reshape(x,[-1]), idx_flattened)
    return y

class ReinforceAgent():	# ACER
    def __init__(self, state_size, action_size):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/stage_3_')
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
	
	# (edit1) for ema
	self.alpha = 0.99	
	
	# (edit1) for q_ret calc
	self.gamma = 0.99

	# (edit1) delta for KL divergence
	self.delta = 1

        #self.model = self.buildModel()
        #self.target_model = self.buildModel()
	
	# (edit1) build actor and critic
	self.actor, self.critic, self.polyak  = self.buildModel()

        #self.updateTargetModel()

        if self.load_model:
            self.model.set_weights(load_model(self.dirPath + str(self.load_episode) + ".h5").get_weights())

            with open(self.dirPath + str(self.load_episode) + '.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

	""" 
	(edit1) change model to actor critic suited for ACER
    
    def buildModel(self):
        model = Sequential()
        dropout = 0.2

        model.add(Dense(64, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))

        model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))

        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()

        return model
	"""



	# (edit1) changed model to A-C
    def buildModel(self):	
	input = Input(shape=(self.state_size,))
        dense1 = Dense(256, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform', name='dense1')(input)
        #dense2 = Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
	pi = Dense(self.action_size, kernel_initializer='lecun_uniform', activation='softmax', name='pi')(dense1)
	# or try...
	#pi = Dense(self.action_size, kernel_initializer='lecun_uniform', activation=tf.nn.softmax)(dense1)
	q = Dense(self.action_size, kernel_initializer='lecun_uniform', name='q')(dense1)
	
	actor = Model(input=[input], output=[pi])
	#actor.compile(optimizer=Adam(lr=self.learning_rate), loss=custom_loss) # need to think about what to do with loss
	#actor.compile(optimizer=Adam(lr=self.learning_rate)) # use GradientTape apply_gradient?

	critic = Model(input=[input], output=[q])
	#critic.compile(optimizer=Adam(lr=self.learning_rate), loss=custom2_loss) # need to thinkg about what to do for loss
	#critic.compile(optimizer=Adam(lr=self.learning_rate))

	polyak = clone_model(actor)	# just preparing for later on
	polyak.set_weights(actor.get_weights()) # start out same as actor

	return actor, critic, polyak
    

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    # (to be editted 1) don't need updateTargetModel since gnna be updated all the time
#    def updateTargetModel(self):
#        self.target_model.set_weights(self.model.get_weights())

    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.actor.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            return np.argmax(q_value[0])

    #(edit1) changed next_state to mu
    def appendMemory(self, state, action, reward, mu, done):
        self.memory.append((state, action, reward, mu, done))

    # :edit1) from OpenAI baseline a2c.utils
    # not verified if it works yet...
    #def get_by_index(x, inx):
#	assert(len(x.get_shape())==2)
#	assert(len(idx.get_shape())==1)
#	idx_flattened = tf.range(0, x.shape[0]) * x.shape[1] + idx
#	y = tf.gather(tf.reshape(x,[-1]), idx_flattened)
#	return

    def update_polyak(self):
	old = [self.alpha*x for x in self.polyak.get_weights()]
	new = [(1-self.alpha)*x for x in self.actor.get_weights()]
	self.polyak.set_weights((old[0]+new[0], old[1]+new[1], old[2]+new[2], old[3]+new[3]))
    
    # (edit1) new training model for ACER
    def trainModel(self, on_policy=False, target=False):
	# sampling 
        if on_policy:
	    mini_batch = [self.memory[-1]]    
	else:
	    mini_batch = random.sample(self.memory, self.batch_size)
	
	states = [x[0] for x in mini_batch]
	actions = [[x[1]] for x in mini_batch]
	rewards = [x[2] for x in mini_batch]
	#next_states = [x[3] for x in mini_batch]
	mus = [x[3] for x in mini_batch]
	dones = [int(x[4]) for x in mini_batch]

	states = np.asarray(states)
	actions = np.asarray(actions)
	mus = np.asarray(mus)



	states = tf.convert_to_tensor(states, dtype=tf.float32)
	actions = tf.convert_to_tensor(actions, dtype=tf.int32)
	mus = tf.convert_to_tensor(mus, dtype=tf.float32)

	q = self.critic(states)
	pi = self.actor(states)
	pi_avg = self.polyak(states)

	#q_a = get_by_index(q, actions) # get_by_index might not work
				       # might need to implement tf.gather another way
	
	#pi_a = get_by_index(pi, actions)

	a_index = tf.stack([tf.range(tf.shape(actions)[0]), actions[:,0]], axis=-1)
	
	q_a = tf.gather_nd(q, a_index)
	pi_a = tf.gather_nd(pi, a_index)

	v = tf.reduce_sum(q * pi, axis=-1) # might need ,axis = -1
	
	rho = pi / (mu + 1e-6)
	rho_a = tf.gather_nd(rho, a_index)
	rho_bar = tf.minimum(1.0, rho_a)
	
	print(v,'\n ############################################## \n', dones)
	q_ret = v[-1] * dones[-1]
	
	q_rets = []

	for i in reversed(range(len(rewards))):
	    q_ret = rewards[i] + self.gamma * q_ret
	    q_rets.append(q_ret)
	    q_ret = (rho_bar[i] * (q_ret - q_a[i])) + v[i]
	    # (edit1?) need correction for when new sequence is beginning ??

	q_rets.reverse()
	#q_ret = tf.reshape(tf.stack(values=q_rets, axis=1), [-1])	# (edit1) in reference to seq_to_batch 
									# OpenAI baseline a2c.utils
	print(q_ret)
	print('#############################################')
	print(q_ret.shape)
	q_ret = tf.expand_dims(tf.convert_to_tensor(q_ret, dtype=tf.float32), axis=1)

	# adv = q_ret - v
	loss_f = - rho_bar * tf.log(pi_a + 1e-6) * (q_ret - v)
	#loss_f = tf.reduce_mean(loss_f)
	loss_bc = - tf.maximum((1-c/rho),0.0) * pi * tf.log(pi) * (q - v)	# note that tf.____ functions might need to be
									# tf.math.____
									# might need to reshape either q or v
	#loss_bc = tf.reduce_mean(loss_bc)
	loss_q = tf.reduce_mean(tf.square(tf.stop_gradient(q_ret) - q_a) * 0.5)

	# (edit1) in reference t

	g = tf.gradients(-(loss_f + loss_bc), pi)
	k = pi_avg / (pi + 1e-6)

	#k_dot_g = tf.reduce_sum(k*g, axis=-1)
	grad_pi = tf.maximum(0.0, (tf.reduce_sum(k*g, axis=-1) - self.delta) / (tf.reduce_sum(tf.square(k), axis=-1) + 1e-6))
	grad_pi = tf.gradient(grad_pi, self.actor.trainable_variables)
	grad_v = tf.gradients(loss_q, self.critic.trainable_variables)

	trainer_pi = tf.train.Adam(learning_rate=self.learning_rate)
	trainer_v = tf.train.Adam(learning_rate=self.learning_rate)

	trainer_pi.apply_gradients(grad_pi)
	trainer_v.apply_gradients(grad_v)

	self.update_polyak()
"""
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)
	# need to edit from below

        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            next_states = mini_batch[i][3]
            dones = mini_batch[i][4]

            q_value = self.model.predict(states.reshape(1, len(states)))
            self.q_value = q_value

            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)))

            else:
                next_target = self.model.predict(next_states.reshape(1, len(next_states)))

            next_q_value = self.getQvalue(rewards, next_target, dones)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = q_value.copy()

            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)

        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)
"""

if __name__ == '__main__':
    rospy.init_node('turtlebot3_dqn_stage_3')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    state_size = 28
    #(edit1) action size 5-->3
    #action_size = 5
    action_size = 3

    env = Env(action_size)

    agent = ReinforceAgent(state_size, action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()

    for e in range(agent.load_episode + 1, EPISODES):
        done = False
        state = env.reset()
        score = 0
        for t in range(agent.episode_step):
            #action = agent.getAction(state)
	    mu = agent.actor.predict(state.reshape(1, len(state)))
	    action = np.random.choice(range(len(mu[0])), p=mu[0])

            next_state, reward, done = env.step(action)
	    
	    # (to be editted1) change next_state to mu
            agent.appendMemory(state, action, reward, mu, done)

            if len(agent.memory) >= agent.train_start:
                if global_step <= agent.target_update:
                    agent.trainModel()
                else:
                    agent.trainModel(True)

            score += reward
            state = next_state
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

            if e % 10 == 0:
                agent.model.save(agent.dirPath + str(e) + '.h5')
                with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                    json.dump(param_dictionary, outfile)

            if t >= 500:
                rospy.loginfo("Time out!!")
                done = True

            if done:
                result.data = [score, np.max(agent.critic.predict(state.reshape(1, len(state)))[0])]
                pub_result.publish(result)
                #agent.updateTargetModel()
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
            if global_step % agent.target_update == 0:
                rospy.loginfo("UPDATE TARGET NETWORK")

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
