import gym
import numpy as np
import matplotlib.pyplot as plt


def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    """ TODO: implement the n-step sarsa algorithm """
    pass


env=gym.make('FrozenLake-v0', map_name="8x8")
# TODO: run multiple times, evaluate the performance for different n and alpha
nstep_sarsa(env)
