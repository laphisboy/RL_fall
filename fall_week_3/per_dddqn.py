import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np


class DuelingDQN(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(DuelingDQN, self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, activation ='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis = 1, keepdims=True)))

        # axis = 1 : average over the action dimension rather than the batch dimension

        return Q

    def advantage(self,state):          # why is this necessary? can't we use def call() to return A as well?
        x= self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype= np.float32)  # *input_shape : unpack list or tuple given?

        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype= np.float32)  # *input_shape : unpack list or tuple given?

        self.state_memory_pad = np.zeros((self.mem_size, *input_shape), dtype = np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
    
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        ### PER 
        self.priority_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.state_memory_pad = np.zeros((self.mem_size, *input_shape), dtype= np.float32)  # *input_shape : unpack list or tuple given?
        self.new_state_memory_pad = np.zeros((self.mem_size, *input_shape), dtype= np.float32)  # *input_shape : unpack list or tuple given?
        self.action_memory_pad = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory_pad = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory_pad = np.zeros(self.mem_size, dtype=np.bool)
        self.priority_memory_pad = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done, priority):
        index = self.mem_cntr % self.mem_size
        #print('index',index)
        #print("self and state")
        #print(self.state_memory[index], self.state_memory[index].shape)
        #print(state, state.shape)
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        self.priority_memory[index] = priority

        self.mem_cntr += 1
        #print("self.mem_cntr" , self.mem_cntr)
    def sample_buffer(self, batch_size):                                ### change to use prioritized experience replay
        max_mem = min(self.mem_cntr, self.mem_size)

        ### implementation of priority sampling

        #print("shapes",self.state_memory.shape, self.new_state_memory.shape, self.priority_memory.shape)

        p_sum = np.sum(self.priority_memory)
        #print("self.priority_mem", self.priority_memory)
        #print("max_mem", max_mem)
        #print("p_sum == 0.0", p_sum, p_sum == 0)        
        if p_sum == 0.0:
            batch = np.random.choice(max_mem, batch_size, replace = False)
        else:
            p = self.priority_memory / p_sum
            p = p[:max_mem]
            vip = np.argwhere(self.priority_memory[:max_mem] == 0).flatten()          ### vip are those that haven't calc error yet
                                                                            ### so upmost priority to calc error
            #print("vip", vip)
            #print("p and p_sum", p, np.sum(p), p_sum == 0)
            batch = np.random.choice(max_mem, batch_size-len(vip), replace=False, p = p)
            batch = np.concatenate((vip, batch))
        ### combine above two for the batch to be sampled

        # replace = False: don't want to sample the same memory twice
        
        #print("batch", max(batch))

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        ### remove the onces sampled because we want to update them
        ### update the priority but to keep track of right order, update all of them
        #print("self.state_mem",self.state_memory.shape)
        #self.state_memory = np.delete(self.state_memory,[batch],0)
        #self.new_state_memory = np.delete(self.new_state_memory,[batch],0)
        #self.action_memory  = np.delete(self.action_memory,[batch])
        #self.reward_memory  = np.delete(self.reward_memory,[batch])
        #self.terminal_memory = np.delete(self.terminal_memory,[batch])
        #self.priority_memory = np.delete(self.priority_memory,[batch])
        
        #print("self.state_mem",self.state_memory.shape)
        #padding = self.state_memory_pad
        #self.state_memory = np.concatenate((self.state_memory, self.state_memory_pad[:batch_size]),axis=0)
        #self.new_state_memory = np.concatenate((self.new_state_memory, self.new_state_memory_pad[:batch_size]),axis=0)
        #self.action_memory = np.concatenate((self.action_memory, self.action_memory_pad[:batch_size]))
        #self.reward_memory = np.concatenate((self.reward_memory, self.reward_memory_pad[:batch_size]))
        #self.terminal_memory = np.concatenate((self.terminal_memory, self.terminal_memory_pad[:batch_size]))
        #self.priority_memory = np.concatenate((self.priority_memory, self.priority_memory_pad[:batch_size]))

        #self.mem_cntr -= batch_size

        return states, actions, rewards, states_, dones, max_mem, batch

    def importance_sampling(self,  max_mem, beta, batch):
        
        p_sum = np.sum(self.priority_memory)
        importance = []
        if p_sum == 0:
            importance =  [1 for i in batch]
        else: 
            for i in batch:
                prob = self.priority_memory[i] / p_sum
                #print("prob", prob)
                importance.append((1/(max_mem*prob))**beta)
        #print("p_sum in importance", p_sum, p_sum == 0.0)
        return importance

    def update_transition(self, states, actions, rewards, states_, dones, priority, batch):
        
        #print("update", batch, priority)
        self.state_memory = np.delete(self.state_memory,batch,0)
        self.new_state_memory = np.delete(self.new_state_memory,batch,0)
        self.action_memory  = np.delete(self.action_memory,batch)
        self.reward_memory  = np.delete(self.reward_memory,batch)
        self.terminal_memory = np.delete(self.terminal_memory,batch)
        self.priority_memory = np.delete(self.priority_memory,batch)

        self.state_memory = np.insert(self.state_memory,batch, states,0)
        self.new_state_memory = np.insert(self.new_state_memory,batch, states,0)
        self.action_memory  = np.insert(self.action_memory,batch, actions)
        self.reward_memory  = np.insert(self.reward_memory,batch, rewards)
        self.terminal_memory = np.insert(self.terminal_memory,batch, dones)
        self.priority_memory = np.insert(self.priority_memory,batch, priority)

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3, eps_end=0.01, mem_size = 1000000,
            fc1_dims = 128, fc2_dims = 128, replace = 100):

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


        self.q_eval.compile(optimizer = Adam(learning_rate = lr), loss = 'mean_squared_error')      ### implemented loss function that works
        self.q_next.compile(optimizer = Adam(learning_rate = lr), loss = 'mean_squared_error')      ### with importance sampling

#        self.q_eval.compile(optimizer = Adam(learning_rate = lr), loss = loss_fn)      ### implemented loss function that works
#        self.q_next.compile(optimizer = Adam(learning_rate = lr), loss = loss_fn)      ### with importance sampling
        
    
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done, 0)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)

        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis = 1).numpy()[0]       # from tensor to numpy array to integer

        return action

    def learn(self, alpha=0.7, beta = 0.5):
        if self.memory.mem_cntr < self.batch_size:
            return

        if self.learn_step_counter % self.replace == 0:
            #print("replace")
            self.q_next.set_weights(self.q_eval.get_weights())

        #print("sample_buffer")
        states, actions, rewards, states_, dones, max_mem , batch= self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval(states)
        q_next = self.q_next(states_)
        
        # changing q_pred doesn't mater because we are passing states to the train function anyway
        
        q_target = q_pred.numpy()
        q_eval = q_next.numpy()

        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)
        error = []
        e = 0.001                       ### a small number to prevent probability from becoming 0
        importance = []
        for idx, terminal in enumerate(dones):
            q_estimate = q_target[idx, actions[idx]]
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma*q_eval[idx, max_actions[idx]]*(1-int(dones[idx]))
            td_error = (q_target[idx, actions[idx]] - q_estimate)
            error.append(td_error)

            ### calc priority update
            priority = (np.abs(td_error)+e)**alpha
            #print("priority", priority)
            ### importance sampling
            #importance.append(self.memory.importance_sampling(priority,max_mem,beta))
            #print("inside",type(states[idx]),states[idx].dtype, states[idx].shape)
            #print(states[idx])
            #print("inside",type(actions[idx]),actions[idx].dtype, actions[idx].shape)
            #print(actions[idx])
            #print("inside",type(rewards[idx]),rewards[idx].dtype, rewards[idx].shape)
            #print(rewards[idx])
            #print("inside",type(states_[idx]),states_[idx].dtype, states_[idx].shape)
            #print(states_[idx])
            #print("inside",type(dones[idx]),dones[idx].dtype, dones[idx].shape)
            #print(dones[idx])
            #self.memory.store_transition(states[idx], actions[idx], rewards[idx], states_[idx], dones[idx], priority)
            
            self.memory.update_transition(states[idx], actions[idx], rewards[idx], states_[idx], dones[idx], priority, batch[idx])

            #print("state[idx]", states[idx].shape)
            #print("q_target", q_target.shape)

            #self.q_eval.fit(states[idx], q_target[idx])
            #self.q_eval.fit(states[idx], q_target, sample_weight = importance)          ### importance included as sample weight
                                                                                           ### so upmost priority to calc error
        #print("before train on batch, states and q_target is ...")
        #print(states.shape)
        #print(q_target.shape)
        #print(len(importance))
        #print(importance)
        #print("going in to training...")
        importance = self.memory.importance_sampling(max_mem, beta, batch)
        #print("importance", importance)        
        self.q_eval.train_on_batch(states, q_target, sample_weight = importance)
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        self.learn_step_counter += 1
        
#    def loss_fn(self, y_true, y_pred):        ### based on MSE for keras
#        y_pred = tf.covert_to_tensor_v2(y_pred)
#        y_true = tf.case(y_true, y_pred.dtype)
#        loss = tf.reduce_mean(tf.square(tf.multiply((y_true - y_pred), importance)))
#        return loss:
