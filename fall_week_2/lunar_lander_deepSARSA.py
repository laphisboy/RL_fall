import gym
from deepSARSA import Agent
from utils import plotLearning
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 64, n_actions = 4, eps_end = 0.01,
            input_dims = [8], lr = 0.001)
    scores, eps_history = [], []
    n_games = 500

    
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        action = agent.choose_action(observation) ###
        while not done:

            observation_, reward, done, info = env.step(action)
            score += reward

            action_ = agent.choose_action(observation) ###
            
            agent.store_transition(observation, action, reward, observation_, action_, done) ###
            agent.learn()
            observation = observation_
            action = action_ ###
            #env.render()

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        
        

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander_deepSARSA_2.png'
    plotLearning(x, scores, eps_history, filename)
