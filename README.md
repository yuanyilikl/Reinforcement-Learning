# Reinforcement-Learning
Through a few small examples to understand the principle of reinforcement learning and its algorithm.

The main application algorithms are:

 Q_Learning, State–action–reward–state–action (SARSA), SARSA(Lambda), DeepQNetwork (DQN)
 
## Environment

* numpy
* pandas
* torch
* time
* TKinter
* sys

## Usage
**1. RL_QL_StraightTreasureGame.py**  
    Set up an environment : o-----T.   It starts at O and ends at T.   
    Algorithms used: QLearning
```
python RL_QL_StraightTreasureGame.py train
```
Parameter Settings :    
    greedy: epsilon = 0.9        
    learning rate:  alpha = 0.1        
    Discount rate: gamma = 0.9        
    run episode: max_episodes = 20      
``` 
python RL_QL_StraightTreasureGame.py train --epsilon=0.9 --alpha=0.1 --gamma=0.9 --max_episodes=30
```
**2. RL_QLandSar_maze.py**  
Environment : a maze (maze_env.py)(From Morvan)  
Algorithms used : QLearning or Sarsa
```python
python RL_QLandSar_maze.py train --method=qlearning 
python RL_QLandSar_maze.py train --method=sarsa 
```
**3. Q_brain_m.py and run_this_m.py**  
Environment : a maze (maze_env.py or maze1_env.py) (From Morvan)  
Algorithms used : QLearning or Sarsa or Sarsa(lambda)
```python
python run_this_m.py train --method=qlearning 
python run_this_m.py train --method=sarsa 
python run_this_m.py train --method=sarsa_lam 
```
**4. DQN and DQN_run.py**  
Environment : a maze (maze1_env.py) (From Morvan)  
Algorithms used : QLearning or Sarsa or Sarsa(lambda)
```python
python run_this_m.py train --max_episodes=500
```
## Reference
[Morvan](https://mofanpy.com/)

[Morvan's github](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents)

[Q-Learning](https://blog.csdn.net/itplus/article/details/9361915), [Sarsa](https://zhuanlan.zhihu.com/p/24860793), [Sarsa(lambda)](https://zhuanlan.zhihu.com/p/74346644), [DQN](https://cloud.tencent.com/developer/article/1004953).