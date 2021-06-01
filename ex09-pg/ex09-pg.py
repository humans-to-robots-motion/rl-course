import gym
import numpy as np
import matplotlib.pyplot as plt


def policy(state, theta):
    """ TODO: return probabilities for actions under softmax action selection """
    return [0.5, 0.5]  # both actions with 0.5 probability => random


def generate_episode(env, theta, display=False):
    """ enerates one episode and returns the list of states, the list of rewards and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)

        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)

    return states, rewards, actions


def REINFORCE(env):
    theta = np.random.rand(4, 2)  # policy parameters

    for e in range(10000):
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, True)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        print("episode: " + str(e) + " length: " + str(len(states)))
        # TODO: keep track of previous 100 episode lengths and compute mean

        # TODO: implement the reinforce algorithm to improve the policy weights





def main():
    env = gym.make('CartPole-v1')
    REINFORCE(env)
    env.close()


if __name__ == "__main__":
    main()
