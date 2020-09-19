# DDDQN on Lunar Lander

Dueling Double Deep Q-Network   
Dueling structure where Q-target network has update delay period that is significantly long  
and Q-target network evaluates while Q-network chooses the argmax next action

## Reference

no forking but basically copied down code shown through youtube  

https://www.youtube.com/watch?v=A39cjchWnsU

## Lunar Lander with DDDQN (Tensorflow.keras)

![lunar_lander_dddqn](https://github.com/laphisboy/RL_fall/blob/master/fall_week_3/lunar_lander_dddqn.png)

When compared with results shown .png below,  
DDDQN shows less noisy convergence to optimal policy (or at least the better policy)  
Note that the exploration rate (epsilon) is different above and below  
which could be reason to why DDDQN seems to converge slower than DQN  


## Lunar Lander with DQN (PyTorch)

Learns faster

![lundar_lander_dqn](https://github.com/laphisboy/RL_fall/blob/master/fall_week_2/lunar_lander_dqn.png)

## Lunar Lander with Deep SARSA (PyTorch)

Has difficulty learning (slower)  

![lunar_lander_deepSARSA](https://github.com/laphisboy/RL_fall/blob/master/fall_week_2/lunar_lander_deepSARSA.png)
