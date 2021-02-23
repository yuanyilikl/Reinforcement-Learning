import numpy as np
import pandas as pd
import time

class Config(object):
    n_states     = 8
    actions      = ["left", "right"]
    epsilon      = 0.9    # greedy
    alpha        = 0.1    # learing rate
    gamma        = 0.9    # Discount rate
    max_episodes = 20

opt = Config()

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns = actions, 
    ) 
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, : ]
    if (np.random.uniform() > opt.epsilon) or (state_actions.all() == 0):
        action_name = np.random.choice(opt.actions)
    else:
        action = state_actions.argmax()
        action_name = 'right' if action else 'left'


    return action_name

def get_env_feedback(S, A):
    if A == "right" :    # move right
        if S == opt.n_states - 2:
            S_ = "terminal"
            R  = 1
        else:
            S_ = S + 1
            R  = 0
    else:                # move left
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    
    return S_, R

def update_env(S, episode, step_counter):

    env_list = ["-"]*(opt.n_states-1) + ["T"]
    if S == "terminal" :
        interaction = "Episode %s: total_steps = %s " %(episode+1, step_counter)
        print("\r{}".format(interaction), end = "\n")
        time.sleep(2)
        print("\r                    ", end = "")

    else:
        env_list[S] = "o"
        interaction = "".join(env_list)
        print("\r{}".format(interaction), end = "")
        time.sleep(1)

def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    q_table = build_q_table(opt.n_states, opt.actions)
    print(q_table)
    for episode in range(opt.max_episodes):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            
            if S_ != "terminal":
                q_target = R + opt.gamma * q_table.iloc[S_, : ].max()
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += opt.alpha * (q_target - q_predict)
            S = S_

            update_env(S, episode, step_counter + 1)

            step_counter += 1

    print(q_table)
    return q_table

if __name__ == '__main__':
    import fire
    fire.Fire()

