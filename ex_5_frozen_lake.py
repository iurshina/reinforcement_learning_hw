import gym
import numpy as np
from collections import defaultdict

env = gym.make("FrozenLake-v0")

alpha = 0.4
gamma = 0.1
epsilon = 0.33


def action(Q, obs):
    if np.random.rand() <= epsilon:
        return np.random.randint(0, env.action_space.n)
    else:
        return np.argmax(Q[obs])


# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3
def get_policy(Q):
    pi = {}
    for k in Q:
        pi[k] = np.argmax(Q[k])
    return pi


def get_v(Q):
    V = {}
    for k in Q:
        V[k] = np.max(Q[k])
    return V


# task 2
def sarsa(steps):
    Q = defaultdict(lambda: [0.9, 0.9, 0.9, 0.9])
    t = 0
    for i in range(steps):
        observation = env.reset()
        act = action(Q, observation)

        prev_obs = None
        prev_act = None

        done = False
        while not done:
            observation, reward, done, info = env.step(act)
            act = action(Q, observation)

            if prev_obs is not None:
                prev_q = Q[prev_obs][prev_act]
                if done:
                    Q[prev_obs][prev_act] = prev_q + alpha * (reward - prev_q)
                else:
                    Q[prev_obs][prev_act] = prev_q + alpha * (reward + gamma * Q[observation][act] - prev_q)

            prev_obs = observation
            prev_act = act

            t += 1

    print "Avg length of an episode:", t / steps

    return Q, get_v(Q), get_policy(Q)


def main():
    Q, V, pi = sarsa(100000)
    print


if __name__ == '__main__':
    main()
