import numpy as np
import torch
from torch import functional
from maze1_env import Maze
from DQN import DeepQNetwork

class Config(object):
    TARGET_REPLACE_ITER = 100   # target update frequency
    BATCH_SIZE  = 32
    LR          = 0.01          # learning rate
    EPSILON     = 0.9           # greedy policy
    GAMMA       = 0.9           # reward discount
    MEMORY_SIZE = 500
    MAX_EPISODE = 500

opt = Config()
env = Maze()

def update_dqn(RL):
    step = 0
    for episode in range(opt.MAX_EPISODE):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()
            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                print('--------episode:{0}-------'.format(episode))
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()

def train(**kwargs):
    for k_, v_ in kwargs.items():
       setattr(opt, k_, v_) 

    RL = DeepQNetwork(env.n_actions, env.n_features, opt)
    env.after(100, update_dqn(RL))
    env.mainloop()
 
if __name__ == '__main__':   
    import fire
    fire.Fire()
