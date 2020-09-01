# Comparing SARSA and Q-learning on just_lake / frozen_lake / cliff

## just_lake

showed no significant difference

![just_lake_SARSA](https://github.com/laphisboy/RL_fall/blob/master/fall_week_1/just_lake_SARSA_screen.png)
![just_lake_q_screen](https://github.com/laphisboy/RL_fall/blob/master/fall_week_1/just_lake_q_screen.png)

## frozen_lake

q-learning showed better performance on frozen_lake  
however seems that q-network or DQN would be more suitable for this  

![frozen_lake_SARSA](https://github.com/laphisboy/RL_fall/blob/master/fall_week_1/frozen_lake_SARSA_screen.png)
![frozen_lake_q](https://github.com/laphisboy/RL_fall/blob/master/fall_week_1/frozen_lake_q_screen.png)

## cliff

as mentioned in Sutton-Barto,  
SARSA showed better performance when limited number of timesteps small enough  
That is because SARSA tends to go for the safe path  
and Q-learning tends to go to the optimal path (also dangerous)  
however with enough training both method will reach optimal policy  

![cliff_SARSA](https://github.com/laphisboy/RL_fall/blob/master/fall_week_1/cliff_SARSA_screen.png)
![cliff_q](https://github.com/laphisboy/RL_fall/blob/master/fall_week_1/cliff_q_screen.png)
