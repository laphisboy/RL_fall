import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical
import torch.multiprocessing as mp
import time

# Hyperparameters
n_train_processes = 4       # 4 train processes
leaning_rate = 0.001        # increased learning rate from 0.0002 to 0.001
update_interval = 5
gamma  = 0.98
max_train_ep = 300
max_test_ep = 500            # increased number of test_ep but removed delay for test

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4,256)                 # shares one layer
                                                    # questionable if 4 --> 256 is really necessary
                                                    # also could think of implementing A3C with no shared network
                                                    # between pi and v
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256,1)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim = softmax_dim)
        return prob

    def v(self,x ):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

def train(global_model, rank):
    local_model = ActorCritic()
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(global_model.parameters(), lr = leaning_rate)

    env = gym.make('CartPole-v1')

    for n_epi in range(max_train_ep):
        done = False
        s = env.reset()

        while not done:
            s_list, a_list, r_list = [], [], []

            for t in range(update_interval):
                # choose action based on policy pi
                prob = local_model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)   # torch.distributions
                                        # use:
                a = m.sample().item()

                # observe action
                s_prime, r, done, info = env.step(a)

                s_list.append(s)
                a_list.append([a])
                r_list.append(r/100.0)

                s = s_prime

                if done:
                    break

            # meaning of s_final?
            s_final = torch.tensor(s_prime, dtype= torch.float)

            R = 0.0 if done else local_model.v(s_final).item()      # terminal or bootstrapping


            # bootstrapping from the last state
            # in reverse order
            td_target_list = []
            for reward in r_list[::-1]:
                R = gamma * R + reward
                td_target_list.append([R])

            td_target_list.reverse()        # back to match order with others

            s_batch, a_batch, td_target = torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), \
                                            torch.tensor(td_target_list)
            advantage = td_target - local_model.v(s_batch)          # R - V(s;theta)

            # leadning to gradient for pi
            pi = local_model.pi(s_batch, softmax_dim=1)
            pi_a = pi.gather(1, a_batch)

            # how is gradient for pi and gradient for v combined here?
            # because designed to share parameters?
            loss = -torch.log(pi_a) * advantage.detach() + \
                    F.smooth_l1_loss(local_model.v(s_batch), td_target.detach())

            optimizer.zero_grad()
            loss.mean().backward()

            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad           # ._grad is writable while .grad is not

            optimizer.step()

            local_model.load_state_dict(global_model.state_dict())

    env.close()
    print("Training process {} reached maximum episode.".format(rank))

def test(global_model):
    env = gym.make('CartPole-v1')
    score = 0.0
    print_interval = 20

    for n_epi in range(max_test_ep):
        done = False
        s = env.reset()

        while not done:
            prob = global_model.pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().item()
            s_prime, r, done, info = env.step(a)
            s = s_prime
            score += r

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode: {}, avg score : {:.1f}".format(n_epi, score/print_interval))

        score = 0.0
        #time.sleep(1)
    env.close()


if __name__ == '__main__':
    global_model = ActorCritic()
    global_model.share_memory()             # preparation for torch.multiprocessing.Process()

    # question
    # is torch.multiprocessing the best for A3c
    # asynchronous

    processes = []

    for rank in range(n_train_processes + 1):
        if rank == 0:
            p = mp.Process(target = test, args=(global_model,))
        else:
            p = mp.Process(target=train, args=(global_model, rank,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


