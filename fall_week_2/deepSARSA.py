import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepSARSA(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepSARSA, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)      # what is the meaning of *... pointing? : unpack list
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, 
            max_mem_size = 100000, eps_end = 0.05, eps_dec = 5e-4):                 # what is eps_end and eps_dec
                                                                                    # probably epsilon minimum
                                                                                    # and how fast epsilon decreases
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = DeepSARSA(lr, n_actions = n_actions, input_dims = input_dims,
                fc1_dims = 256, fc2_dims = 256)

        self.Q_next = DeepSARSA(lr, n_actions = n_actions, input_dims = input_dims, 
                fc1_dims = 64, fc2_dims = 64)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype = np.int32)
        self.next_action_memory = np.zeros(self.mem_size, dtype = np.int32)         # add numpy array for next action
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)

    def store_transition(self, state, action, reward, state_, action_, terminal):   # take account of next action when storing
        index = self.mem_cntr % self.mem_size
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_                               # s_prime
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.next_action_memory[index] = action_
        self.terminal_memory[index] = terminal

        self.mem_cntr +=  1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)          # input and get ready?
            actions = self.Q_eval.forward(state)                            # get action probs from network
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()                                   # initiallize grad

        max_mem = min(self.mem_cntr, self.mem_size)                         # to ensure batch to hold whats filled up

        batch = np.random.choice(max_mem, self.batch_size, replace = False) # not really sure what's going on from here.... to...
                                                                            # just array of random numbers?
                                                                            # pikcing random numbers 
                                                                            
        batch_index = np.arange(self.batch_size, dtype = np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        next_action_batch = self.next_action_memory[batch]                  ### added
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]            # update the values of actions actually took
        q_next = self.Q_eval.forward(new_state_batch)[batch_index, next_action_batch]   ### also use the actions actually took in next state
        q_next[terminal_batch] = 0.0                                        # if done then done do forward passing

        q_target = reward_batch + self.gamma*q_next                         ### Q value of action actually took
                                                                            # max returns value and index so [0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                else self.eps_min

        # if self.iter_cntr % self.replace_target == 0:
        #    self.Q_next.load_state_dict(self.Q_eval.state_dict())

