import torch 
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(n_states, 100),  
            nn.ReLU(True),
            nn.Linear(100, 50),  
            nn.ReLU(True),
            nn.Linear(50, n_actions),                    
            )
    def forward(self, input):
        return self.main(input)

class DeepQNetwork(object):
    def __init__(self, n_actions, n_states, opt):
        self.learn_step_counter  = 0
        self.target_replace_iter = opt.TARGET_REPLACE_ITER                                    
        self.memory_size = opt.MEMORY_SIZE 
        self.epsilon     = opt.EPSILON        
        self.batch_size  = opt.BATCH_SIZE       
        self.gamma       = opt.GAMMA 
        self.eval_net    = Net(n_states, n_actions)
        self.target_net  = Net(n_states, n_actions)                                       
        self.memory      = np.zeros((self.memory_size, n_states * 2 + 2))     
        self.optimizer   = torch.optim.Adam(self.eval_net.parameters(), lr = opt.LR)
        self.loss_func   = nn.MSELoss()
        self.n_actions   = n_actions
        self.n_states    = n_states


    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        
        # input only one sample
        if np.random.uniform() < self.epsilon:   
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
        else:  
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter  += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.memory_size, self.batch_size)
        b_memory     = self.memory[sample_index, :]
        b_s  = torch.FloatTensor(b_memory[:, : self.n_states])
        b_a  = torch.LongTensor(b_memory[:, self.n_states : self.n_states+1].astype(int))
        b_r  = torch.FloatTensor(b_memory[:, self.n_states + 1 : self.n_states + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states : ])

        # q_eval w.r.t the action in experience
        q_eval   = self.eval_net(b_s ).gather(1, b_a)                             # shape (batch, 1)
        q_next   = self.target_net(b_s_).detach()                                 # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss     = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

