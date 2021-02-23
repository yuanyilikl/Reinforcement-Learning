from maze_env import Maze
from Q_brain import LEARNING

class Config(object):
    epsilon      = 0.9    # e_greedy
    alpha        = 0.1    # learing rate
    gamma        = 0.9    # reward_decay
    trace_decay  = 0.9
    max_episodes = 20
    method       = 'qlearning' # method = 'qlearning', 'sarsa', or 'sarsa_lam', 'qlearning' is default


opt = Config()
env = Maze()

def update_q(RL):
    for episode in range(opt.max_episodes):
        observation = env.reset()

        while True:
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            RL.learn_q(str(observation), action, reward, str(observation_))
            observation = observation_
            if done:
                print('--------episode:{0}-------'.format(episode))
                break

    print("game over")
    env.destroy()

def update_sarsa(RL):
    for episode in range(opt.max_episodes):
        observation = env.reset()
        print(type(observation))
        action = RL.choose_action(str(observation))

        while True:
            env.render()
            observation_, reward, done = env.step(action)
            action_ = RL.choose_action(str(observation_))
            RL.learn_sarsa(str(observation), action, reward, str(observation_), action_)
            action = action_
            observation = observation_
            if done:
                print('--------episode:{0}-------'.format(episode))
                break

    print("game over")
    env.destroy()

def update_sarsa_lam(RL):
    for episode in range(opt.max_episodes):
        observation = env.reset()
        action = RL.choose_action(str(observation))

        while True:
            env.render()
            observation_, reward, done = env.step(action)
            action_ = RL.choose_action(str(observation_))
            RL.learn_sarsa_lam(str(observation), action, reward, str(observation_), action_)
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

    RL = LEARNING(opt, actions = list(range(env.n_actions))) 

    ### there has 3 kinds of traindition way 
    if opt.method == 'qlearning':
        env.after(100, update_q(RL))
    if opt.method == 'sarsa':
        env.after(100, update_sarsa(RL))
    if opt.method == 'sarsa_lam':
        env.after(100, update_sarsa_lam(RL))

    env.mainloop()
 
if __name__ == '__main__':   
    import fire
    fire.Fire()
