import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def generate_demonstrations(env, expertpolicy, epsilon=0.1, n_trajs=100):
    """ This is a helper function that generates trajectories using an expert policy """
    demonstrations = []
    for d in range(n_trajs):
        traj = []
        state = env.reset()
        for i in range(100):
            if np.random.uniform() < epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = expertpolicy[state]
            traj.append((state, action))  # one trajectory is a list with (state, action) pairs
            state, _, done, info = env.step(action)
            if done:
                traj.append((state, 0))
                break
        demonstrations.append(traj)
    return demonstrations  # return list of trajectories


def plot_rewards(rewards, env):
    """ This is a helper function to plot the reward function"""
    fig = plt.figure()
    dims = env.desc.shape
    plt.imshow(np.reshape(rewards, dims), origin='upper', 
               extent=[0,dims[0],0,dims[1]], 
               cmap=plt.cm.RdYlGn, interpolation='none')
    for x, y in product(range(dims[0]), range(dims[1])):
        plt.text(y+0.5, dims[0]-x-0.5, '{:.3f}'.format(np.reshape(rewards, dims)[x,y]),
                horizontalalignment='center', 
                verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def value_iteration(env, rewards):
    """ Computes a policy using value iteration given a list of rewards (one reward per state) """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V_states = np.zeros(n_states)
    theta = 1e-8
    gamma = .9
    maxiter = 1000
    policy = np.zeros(n_states, dtype=np.int)
    for iter in range(maxiter):
        delta = 0.
        for s in range(n_states):
            v = V_states[s]
            v_actions = np.zeros(n_actions) # values for possible next actions
            for a in range(n_actions):  # compute values for possible next actions
                v_actions[a] = rewards[s]
                for tuple in env.P[s][a]:  # this implements the sum over s'
                    v_actions[a] += tuple[0]*gamma*V_states[tuple[1]]  # discounted value of next state
            policy[s] = np.argmax(v_actions)
            V_states[s] = np.max(v_actions)  # use the max
            delta = max(delta, abs(v-V_states[s]))

        if delta < theta:
            break

    return policy




def main():
    env = gym.make('FrozenLake-v0')
    #env.render()
    env.seed(0)
    np.random.seed(0)
    expertpolicy = [0, 3, 0, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
    trajs = generate_demonstrations(env, expertpolicy, 0.1, 20)  # list of trajectories
    print("one trajectory is a list with (state, action) pairs:")
    print (trajs[0])




if __name__ == "__main__":
    main()
