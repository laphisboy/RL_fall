from kaggle_environments import evaluate, make, utils

env = make("connectx", debug = True)
env.render()

def my_agent(observation, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c]==0])

env.reset()

env.run([my_agent, "random"])
env.render(mode="ipython", width = 500, height = 450)
