from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True

for step in range(1000):
    if done:
        state = env.reset()

    state, reward, done, info = env.step(env.action_space.sample())
    print("state size", state.shape)
    # print("state values", state)
    print("reward", reward)
    print("done", done)
    print("info", info)

    env.render()

env.close()
