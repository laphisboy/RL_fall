import gym
import numpy as np
from dddqn import Agent
from utils import plotLearning

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(lr = 0.0005, gamma = 0.99, n_actions = 4, epsilon = 1.0, batch_size = 64, input_dims =[8])
    
    n_games = 500

    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0 
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)

            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score,  'avg score %.1f' % avg_score)

    filename = 'lunar_lander_dddqn.png'
    x = [i + 1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)


