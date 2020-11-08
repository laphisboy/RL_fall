import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np
import gc

class DuelingDQN(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(DuelingDQN, self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):  # call: just calling the name of the object we can operate this function
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        # axis = 1 : average over the action dimension rather than the batch dimension

        return Q

    def advantage(self, state):  # why is this necessary? can't we use def call() to return A as well?
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A


class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)  # *input_shape : unpack list or tuple given?

        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)  # *input_shape : unpack list or tuple given?

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        ### added
        self.priority_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done, priority):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        ### added
        self.priority_memory[index] = priority

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        ### added and changed
        p_sum = np.sum(self.priority_memory)
        if p_sum == 0.0:
            batch = np.random.choice(max_mem, batch_size, replace=False)
        else:
            p = self.priority_memory / p_sum
            p = p[:max_mem]
            vip = np.argwhere(self.priority_memory[:max_mem] == 0).flatten()
            ### vip is for experience without TD error calc

            batch = np.random.choice(max_mem, batch_size - len(vip), replace=False, p=p)
            batch = np.concatenate((vip,batch))         ### since vip has p = 0, no chance of overlap

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones, max_mem, batch

    def importance_sampling(self,max_mem, beta, batch):

        beta_to_use = min(1, beta+self.mem_cntr/100000)     ### need beta to increase up to 1

        p_sum = np.sum(self.priority_memory)
        importance = []

        if p_sum == 0:      ### if no prioritized experience replay yet
            importance = [1 for i in batch]

        else:
            for i in batch:
                prob = self.priority_memory[i] / p_sum
                importance.append((1/(max_mem*prob))**beta_to_use)

        return importance

    def update_transition(self, states, actions, rewards, states_, dones, priority, batch):

        ### update by delete and replace

        self.state_memory = np.delete(self.state_memory, batch, 0)
        self.new_state_memory = np.delete(self.new_state_memory, batch, 0)
        self.action_memory = np.delete(self.action_memory, batch)
        self.reward_memory = np.delete(self.reward_memory, batch)
        self.terminal_memory = np.delete(self.terminal_memory, batch)
        self.priority_memory = np.delete(self.priority_memory, batch)

        self.state_memory = np.insert(self.state_memory, batch, states, 0)
        self.new_state_memory = np.insert(self.new_state_memory, batch, states_, 0)
        self.action_memory = np.insert(self.action_memory, batch, actions)
        self.reward_memory = np.insert(self.reward_memory, batch, rewards)
        self.terminal_memory = np.insert(self.terminal_memory, batch, dones)
        self.priority_memory = np.insert(self.priority_memory, batch, priority)


class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3, eps_end=0.01,
                 mem_size=20000,
                 fc1_dims=128, fc2_dims=128, replace=100):

    ### reduced mem_size for faster computation...
    ### but original mem_size of 10^6 is given in paper T.T
    ### thinking of increasing replace to have a more double DQN character for longer timesteps

        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DuelingDQN(n_actions, fc1_dims, fc2_dims)
        self.q_next = DuelingDQN(n_actions, fc1_dims, fc2_dims)

        self.q_eval.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
        self.q_next.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done, 0)
        ### new observation will have priority = 0 which will put it into vip classification

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)

        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]  # from tensor to numpy array to integer

        return action

    def learn(self, alpha =0.6 , beta = 0.4):           ### parameters alpha and beta as given in paper
        if self.memory.mem_cntr < self.batch_size:
            return

        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones, max_mem, batch = self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval(states)
        q_next = self.q_next(states_)

        # changing q_pred doesn't mater because we are passing states to the train function anyway

        q_target = q_pred.numpy()
        q_eval = q_next.numpy()

        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)

        ### calculating priority and importance sampling will be mixed from below
        e = 0.00001 ### small value for preventing priority of 0 after calc.
        importance = []

        for idx, terminal in enumerate(dones):
            q_estimate = q_target[idx, actions[idx]]        ### need to keep estimate for calc TD error
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma * q_eval[idx, max_actions[idx]] * (1 - int(dones[idx]))
            td_error = (q_target[idx, actions[idx]] - q_estimate)

            priority = (np.abs(td_error)+e)**alpha

            ### can update once priority is calc.
            self.memory.update_transition(states[idx], actions[idx], rewards[idx], states_[idx], dones[idx], priority, batch[idx])

        ### importance sampling
        importance = self.memory.importance_sampling(max_mem, beta, batch)
        self.q_eval.train_on_batch(states, q_target, sample_weight = importance)

        #self.q_eval.train_on_batch(states, q_target)
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        self.learn_step_counter += 1

        gc.collect()
