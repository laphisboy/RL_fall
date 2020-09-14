import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen = buffer_limit)   # deque = array that pops front and back

    def put(self, transition):
        self.buffer.append(transition)                          # suspect it is just adding item to buffer

    def sample(self, n):                                        # picks item from buffer by random which
        mini_batch = random.sample(self.buffer, n)              # is random replaybuffer

        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_prime_list.append(s_prime)
            done_mask_list.append([done_mask])

# returning in format of torch tensor for deep calc

        return torch.tensor(s_list, dtype = torch.float), torch.tensor(a_list), \
                torch.tensor(r_list), torch.tensor(s_prime_list, dtype = torch.float), \
                torch.tensor(done_mask_list)


    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    
    def __init__(self):
        super(Qnet, self).__init__()        # what is the purpose of this exactly
                                            # inherit nn.Module --> Qnet is done but super().__init__() isnt it?
                                            # why do we even need def __init__(self) to begin with?
        
# Linear Network of 4 dimension input --> 128 neural network --> left or right next action
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)             # output for action given observation of previous action
        coin = random.random()              # random component for e-greedy

        if coin < epsilon:
            return random.randint(0,1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer, epsilon):
    for i in range(10):
        
        s,a,r,s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)                                             # torch.gather()
                                                                            # gets the value specified by 1, a
                                                                            # which is Q value for given action
                                                                            # gather(dim, index)
        a_prime = q.sample_action(s_prime, epsilon)     ### also need to pick next action according to policy
        q_target_out = q_target(s_prime)
        q_target_a_prime = q_target_out.gather(1,a_prime)

        target = r + gamma * q_target_a_prime * done_mask                        # target y .. or j

        loss = F.smooth_l1_loss(q_a, target)                                # compare y withe actual Q value being calculated

# just training
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    
    q = Qnet()
    q_target = Qnet()

    q_target.load_state_dict(q.state_dict())                        # Initiallize by making these two identical

    memory = ReplayBuffer()                                         # Initiallize replay memory


    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)        # optimize the Qnet() neural network parameters


    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))                # epsilon decreses from 0.08 to 0.01

        s = env.reset()                                             # returns inital observation that describes initial state (?check)
        done = False


        a = q.sample_action(torch.from_numpy(s).float(), epsilon) # Note that s = env
                                                                    # thus torch.from_numpy(s) : a, r, s_prime, done_mask?
        while not done:

            s_prime, r, done, info = env.step(a)
            a_prime = q.sample_action(torch.from_numpy(s_prime).float(), epsilon) ### also need to pick next action according to policy

            done_mask = 0.0 if done else 1.0                        # done_mask?? ??
                                                                    # oh done mask is used to dictate whether
                                                                    # forward pass should be included in target or not
                                                                    # (shouldn't if at terminate stage)

            memory.put((s,a,r/100.0, s_prime, done_mask))           # append to Replay Memory

            s = s_prime                                             # next state becomes current state
            a = a_prime                                             ### next action become current action

            score += r                                              # accumulate reward

            if done: break
            
        if memory.size()>2000:                                      # hmm why train when episode terminates?
            train(q, q_target, memory, optimizer, epsilon)                   # shouldn't it be after each episode?
                                                                    # only applied to CartPole-v1 for efficiency
                                                                    # and since the prob is simple?

        if n_epi%print_interval == 0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())                # make sure Q and Q target end as same network
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()

