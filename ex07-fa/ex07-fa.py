import gym
import numpy as np
import matplotlib.pyplot as plt


def random_episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        env.render()
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("reward: ", reward)
        print("")
        if done:
            break


def main():
    env = gym.make('MountainCar-v0')
    env.reset()
    random_episode(env)
    env.close()


if __name__ == "__main__":
    main()
