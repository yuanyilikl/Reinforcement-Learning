import numpy as np
import pandas as pd

class ReinforceLearning(object):
    def __init__(self, opt, actions):
        self.actions = actions
        self.epsilon = opt.epsilon
        self.q_table = pd.DataFrame(columns = actions, dtype = np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() > self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action
 

class LEARNING(ReinforceLearning):
    def __init__(self, opt, actions):
        super(LEARNING, self).__init__(opt, actions)
        self.gamma   = opt.gamma
        self.alpha   = opt.alpha
        self.lambda_ = opt.trace_decay
        self.eligibility_trace = self.q_table.copy()

    def learn_q(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)
   
    def learn_sarsa(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)
           
    def learn_sarsa_lam(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        error = q_target - q_predict

        # Method 1:
        # self.eligibility_trace.loc[s, a] += 1

        # Method 2:
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        self.q_table += self.alpha * error * self.eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_




