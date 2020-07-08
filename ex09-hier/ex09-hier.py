from fourrooms import FourRooms
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp, expit


class SigmoidTermination():
    """ Sigmoid for option termination """
    def __init__(self, nstates):
        self.nstates = nstates
        self.weights = np.zeros((nstates,))

    def pmf(self, state):
        return expit(self.weights[state])

    def sample(self, state):
        return int(np.random.uniform() < self.pmf(state))

    def gradient(self, state):
        return self.pmf(state) * (1. - self.pmf(state))


class SoftmaxPolicy():
    """ Softmax policy to select intra-option primitive actions"""
    def __init__(self, nstates, nactions, temperature=1.0):
        self.nstates = nstates
        self.nactions = nactions
        self.temperature = temperature
        self.weights = np.zeros((nstates, nactions))

    def pmf(self, state):
        exponent = self.weights[state,:] / self.temperature
        return np.exp(exponent - logsumexp(exponent))

    def sample(self, state):
        return int(np.random.choice(self.nactions, p=self.pmf(state)))

    def loggradient(self, state, action):
        g = np.zeros((self.nstates, self.nactions))
        g[state, :] -= self.pmf(state)
        g[state, action] += 1
        return g


def policy_options(state, Q_omega, epsilon=0.1):
    """ Epsilon-greedy policy used to select options """
    if np.random.uniform() < epsilon:
        return np.random.choice(range(Q_omega.shape[1]))
    else:
        return np.argmax(Q_omega[state])


def plot_termination_maps(env, terminations):
    """ Helper function for visualization of option terminations """
    termination_maps = [env.occupancy.astype('float64') for _ in range(len(terminations))]
    for option in range(len(terminations)):
        state = 0
        for i in range(13):
            for j in range(13):
                if termination_maps[option][i,j] == 0:
                    termination_maps[option][i,j] = terminations[option].pmf(state)
                    state += 1

    for o_n, t in enumerate(termination_maps):
        plt.imshow(t, cmap='Blues')
        plt.axis('off')
        plt.title("Option " + str(o_n) + " Termination")
        plt.show()


def option_critic(env):
    noptions = 4  # number of options
    nepisodes = 1000  # number of episodes
    nsteps = 1000  # max number of steps per episode
    
    gamma = 0.99  # discount factor
    lr_term = 0.3  # learning rate for terminations
    lr_intra = 0.3  # learning rate for intra option policy
    lr_critic = 0.5  # learning rate for critic

    temperature = 1e-2  # softmax with temperature (Boltzmann distribution)

    nstates = env.observation_space.shape[0]
    nactions = env.action_space.shape[0]

    history = np.zeros(nepisodes)

    # for each option we create a softmax policy and a sigmoid termination function:
    option_policies = [SoftmaxPolicy(nstates, nactions, temperature) for _ in range(noptions)]
    option_terminations = [SigmoidTermination(nstates) for _ in range(noptions)]

    # tabular Q functions
    Q_omega = np.zeros((nstates, noptions))
    Q_U = np.zeros((nstates, noptions, nactions))

    for episode in range(nepisodes):
        state = env.reset()
        option = policy_options(state, Q_omega)  # select option epsilon greedy

        for step in range(nsteps):
            action = option_policies[option].sample(state)  # select primitive action from option
            nextstate, reward, done, _ = env.step(action)  # perform action: observe nextstate, reward

            # TODO: 1. Options evaluation


            # TODO: 2. Options improvement
            # policies:
            # you can access the weights using: option_policies[option].weights and the log gradient using option_policies[option].loggradient(state, action)
            # terminations:
            # you can access the weights using: option_terminations[option].weights and the gradient using option_terminations[option].gradient(state)
            
            # when option terminates we select a new option
            if option_terminations[option].sample(nextstate):
                option = policy_options(nextstate, Q_omega)

            state = nextstate  # update state

            if done:
                break  # episode ends

        history[episode] = step
        print (episode, step)

    plt.plot(history)
    plt.show()
    plot_termination_maps(env, option_terminations)




def main():
    env = FourRooms()
    state = env.reset()
    print(state)
    option_critic(env)



if __name__ == "__main__":
    main()
