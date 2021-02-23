import numpy as np
import pandas as pd
import random
from maze_env import Maze

class Config(object):
    epsilon      = 0.9    # e_greedy
    alpha        = 0.1    # learing rate
    gamma        = 0.9    # reward_decay
    max_episodes = 20
    method       = 'qlearning' # method = 'qlearning', 'sarsa', 'qlearning' is default

opt = Config()
env = Maze()

class QLearningTable:
    def __init__(self, actions, opt):
        self.actions = actions
        self.epsilon = opt.epsilon
        self.alpha   = opt.alpha
        self.gamma   = opt.gamma
        self.q_table = pd.DataFrame(columns = actions, dtype = np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() > self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action
    
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )

class Sarsa:
    def __init__(self, actions, opt):
        self.actions = actions
        self.epsilon = opt.epsilon
        self.alpha   = opt.alpha
        self.gamma   = opt.gamma
        self.q_table = pd.DataFrame(columns = actions, dtype = np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() > self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action
    
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )

def update_q(QL):
    for episode in range(opt.max_episodes):
        observation = env.reset()

        while True:
            env.render()
            action = QL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            QL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            if done:
                print('--------episode:{0}-------'.format(episode))
                break

    print("game over")
    env.destroy()
 
def update_s(SA):

    for episode in range(opt.max_episodes):
        observation = env.reset()
        action = SA.choose_action(str(observation))

        while True:
            env.render()
            observation_, reward, done = env.step(action)
            action_ = SA.choose_action(str(observation_))
            SA.learn(str(observation), action, reward, str(observation_), action_)
            action = action_
            observation = observation_
            if done:
                print('--------episode:{0}-------'.format(episode))
                break

    print("game over")
    env.destroy()

def train(**kwargs):
    for k_, v_ in kwargs.items():
       setattr(opt, k_, v_)
    
    QL = QLearningTable(actions = list(range(env.n_actions)), opt = opt)
    SA = Sarsa(actions = list(range(env.n_actions)), opt = opt)
    
    if opt.method == 'qlearning':
        env.after(100, update_q(QL))
    if opt.method == 'sarsa':
        env.after(100, update_s(SA))
    env.mainloop()

if __name__ == '__main__':
    import fire
    fire.Fire()
