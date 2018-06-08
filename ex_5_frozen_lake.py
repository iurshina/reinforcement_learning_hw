import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


alpha = 0.4
gamma = 0.99
epsilon = 0.33


def action(Q, obs, space_n):
    if np.random.rand() <= epsilon:
        return np.random.randint(0, space_n)
    else:
        return np.argmax(Q[obs])


# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3
def get_policy(Q):
    pi = {}
    for i in range(0, len(Q)):
        pi[i] = np.argmax(Q[i])
    return pi


def get_v(Q):
    V = {}
    for i in range(0, len(Q)):
        V[i] = np.max(Q[i])
    return V


def expectation(q, is_slippery):
    p_for_e = epsilon / len(q)
    slip_p = 0
    if is_slippery:
        slip_p = 0.33
    max_p = 1 - epsilon + p_for_e - slip_p
    others_p = p_for_e + (slip_p / 3)

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
    t = 0
    for i in range(steps):
        observation = env.reset()
        act = action(Q, observation, env.action_space.n)

        prev_obs = None
        prev_act = None

        done = False
        while not done:
            observation, reward, done, info = env.step(act)
            act = action(Q, observation, env.action_space.n)

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


# task 3
def q_learning(env, steps):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    t = 0
    for i in range(steps):
        observation = env.reset()
        act = action(Q, observation, env.action_space.n)

        prev_obs = None
        prev_act = None

        done = False
        while not done:
            observation, reward, done, info = env.step(act)
            act = action(Q, observation, env.action_space.n)

            if prev_obs is not None:
                prev_q = Q[prev_obs][prev_act]
                if done:
                    Q[prev_obs][prev_act] = prev_q + alpha * (reward - prev_q)
                else:
                    Q[prev_obs][prev_act] = prev_q + alpha * (reward + gamma * np.max(Q[observation]) - prev_q)

            prev_obs = observation
            prev_act = act

            t += 1

    print "Avg length of an episode:", t / steps

    return Q, get_v(Q), get_policy(Q)


# task 5
def expected_sarsa(env, steps, is_slippery):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    t = 0
    for i in range(steps):
        observation = env.reset()
        act = action(Q, observation, env.action_space.n)

        prev_obs = None
        prev_act = None

        done = False
        while not done:
            observation, reward, done, info = env.step(act)
            act = action(Q, observation, env.action_space.n)

            if prev_obs is not None:
                prev_q = Q[prev_obs][prev_act]
                if done:
                    Q[prev_obs][prev_act] = prev_q + alpha * (reward - prev_q)
                else:
                    Q[prev_obs][prev_act] = prev_q + alpha * (reward + gamma * expectation(Q[observation], is_slippery) - prev_q)

            prev_obs = observation
            prev_act = act

            t += 1

    print "Avg length of an episode:", t / steps

    return Q, get_v(Q), get_policy(Q)


# todo: draw everything
def main():
    env = FrozenLakeEnv(is_slippery=True)

    # doesn't really work
    Q, V, pi = sarsa(env, 10000)
    Q, V, pi = q_learning(env, 10000)

    Q, V, pi = expected_sarsa(env, 10000, True)

    # deterministic works
    env = FrozenLakeEnv(is_slippery=False)

    Q, V, pi = sarsa(env, 10000)
    Q, V, pi = q_learning(env, 10000)

    Q, V, pi = expected_sarsa(env, 10000, False)

    print


if __name__ == '__main__':
    main()
