import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import matplotlib.pyplot as plt


# SFFF
# FHFH
# FFFH
# HFFG

# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3


epsilon = 0.1
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor


def action(Q, obs, space_n):
    if np.random.rand() <= epsilon:
        return np.random.randint(0, space_n)
    else:
        return np.argmax(Q[obs])


def get_policy(Q):
    return Q.argmax(axis=1)


def get_v(Q):
    V = [0] * 16
    for i in range(0, len(Q)):
        V[i] = np.max(Q[i])
    return V


def plot_value_func(value_func, title):
    a = np.reshape(value_func, (4, 4))

    a = np.flip(a, 0)

    heatmap = plt.pcolor(a)

    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.4f' % a[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     )

    plt.title(title)
    plt.colorbar(heatmap)
    plt.show()


def action_by_num(num):
    if num == 0:
        return "LEFT"
    if num == 1:
        return "DOWN"
    if num == 2:
        return "RIGHT"
    if num == 3:
        return "UP"


def plot_policy(policy, title):
    a = np.ones((4, 4))
    p = np.reshape(policy, (4, 4))
    p = np.flip(p, 0)
    heatmap = plt.pcolor(a)

    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            plt.text(x + 0.5, y + 0.5, action_by_num(p[y, x]),
                     horizontalalignment='center',
                     verticalalignment='center',
                     )

    plt.colorbar(heatmap)
    plt.title(title)
    plt.show()


def expectation(q, is_slippery):
    # p_for_e = epsilon / len(q)
    slip_p = 0
    if is_slippery:
        slip_p = 0.33
    max_p = 1 - slip_p # - epsilon + p_for_e
    others_p = (slip_p / 3) # + p_for_e

    max = np.max(q)

    E = max * max_p
    max_skipped = False
    for qq in q:
        if qq == max and not max_skipped:
            max_skipped = True
            continue

        E += qq * others_p

    return E


# task 2
def sarsa(env, steps):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    for i in range(len(Q)):
        if i not in (5, 7, 11, 12, 15):
            Q[i] = [1, 1, 1, 1]

    t = 0
    for i in range(steps):
        prev_obs = env.reset()
        prev_act = action(Q, prev_obs, env.action_space.n)

        done = False
        while not done:
            observation, reward, done, info = env.step(prev_act)
            act = action(Q, observation, env.action_space.n)

            prev_q = Q[prev_obs][prev_act]
            Q[prev_obs][prev_act] = prev_q + alpha * (reward + gamma * Q[observation][act] - prev_q)

            prev_obs = observation
            prev_act = act

            t += 1

    print "Avg length of an episode:", t / steps

    return Q, get_v(Q), get_policy(Q)


# task 3
def q_learning(env, steps):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    for i in range(len(Q)):
        if i not in (5, 7, 11, 12, 15):
            Q[i] = [1, 1, 1, 1]

    t = 0
    for i in range(steps):
        prev_obs = env.reset()
        prev_act = action(Q, prev_obs, env.action_space.n)

        done = False
        while not done:
            observation, reward, done, info = env.step(prev_act)
            act = action(Q, observation, env.action_space.n)

            prev_q = Q[prev_obs][prev_act]
            Q[prev_obs][prev_act] = prev_q + alpha * (reward + gamma * np.max(Q[observation]) - prev_q)

            prev_obs = observation
            prev_act = act

            t += 1

    print "Avg length of an episode:", t / steps

    return Q, get_v(Q), get_policy(Q)


# task 5
def expected_sarsa(env, steps, is_slippery):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    for i in range(len(Q)):
        if i not in (5, 7, 11, 12, 15):
            Q[i] = [1, 1, 1, 1]

    t = 0
    for i in range(steps):
        prev_obs = env.reset()
        prev_act = action(Q, prev_obs, env.action_space.n)

        done = False
        while not done:
            observation, reward, done, info = env.step(prev_act)
            act = action(Q, observation, env.action_space.n)

            prev_q = Q[prev_obs][prev_act]
            Q[prev_obs][prev_act] = prev_q + alpha * (reward + gamma * expectation(Q[observation], is_slippery) - prev_q)

            prev_obs = observation
            prev_act = act

            t += 1

    print "Avg length of an episode:", t / steps

    return Q, get_v(Q), get_policy(Q)


def main():
    # non-deterministic
    env = FrozenLakeEnv(is_slippery=True)

    Q, V, pi = sarsa(env, 2000)
    plot_value_func(V, "value func: sarsa - non-determ")
    plot_policy(pi, "policy: sarsa - non-determ")
    # print "pi1", pi
    print "Q1:", Q

    Q, V, pi = q_learning(env, 2000)
    plot_value_func(V, "value func: q-learning - non-determ")
    plot_policy(pi, "policy: q-learning - non-determ")
    # print "pi2", pi
    print "Q2:", Q
    #
    Q, V, pi = expected_sarsa(env, 2000, True)
    plot_value_func(V, "value func: expected sarsa - non-determ")
    plot_policy(pi, "policy: expected sarsa - non-determ")
    # print "pi3", pi
    print "Q3:", Q

    # deterministic
    env = FrozenLakeEnv(is_slippery=False)

    Q, V, pi = sarsa(env, 1000)
    plot_value_func(V, "value func: sarsa - determ")
    plot_policy(pi, "policy: sarsa - determ")
    # print "pi4", pi
    print "Q4:", Q

    Q, V, pi = q_learning(env, 1000)
    plot_value_func(V, "value func: q-learning - determ")
    plot_policy(pi, "policy: q-learning - determ")
    # print "pi5", pi
    print "Q5:", Q

    Q, V, pi = expected_sarsa(env, 1000, False)
    plot_value_func(V, "value func: expected sarsa - determ")
    plot_policy(pi, "policy: expected sarsa - determ")
    # print "pi6", pi
    print "Q6:", Q


if __name__ == '__main__':
    main()
