import gym
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly as py
import plotly.graph_objs as go

env = gym.make('Blackjack-v0')


# task 2
def first_visit_mc_prediction(steps):
    all_returns = defaultdict(list)
    for i in range(1, steps):
        obs = env.reset()
        done = False

        returns = {}
        while not done:
            player_sum, dealer_card, useable_ace = obs
            returns[obs] = 0

            obs, reward, done, _ = env.step(0 if player_sum >= 20 else 1)

            for k, v in returns.iteritems():
                returns[k] = v + reward

        for k, v in returns.iteritems():
            all_returns[k].append(v)

    V = defaultdict(int)
    for k, v in all_returns.iteritems():
        V[k] = sum(v) / len(v)

    return V


# task 3
def safe_div(a, b):
    return 0. if b == 0 else 1.0 * a / b


def mc_es(steps):
    returns = defaultdict(lambda: (0, 0))
    pi = defaultdict(lambda: np.random.randint(2))

    for i in range(steps):
        state = env.reset()

        done = False
        episode = []
        total = 0

        while not done:
            if len(episode) == 0:
                action = np.random.randint(2)
            elif state[0] <= 11:
                action = 1
            else:
                action = pi[state]

            state_new, reward, done, _ = env.step(action)

            if state[0] > 11:
                episode.append((state, action, total))
                state = state_new

            total += reward

        was_seen = {}
        for (state, action, before) in episode:
            key = (state, action)
            if key not in was_seen:
                was_seen[key] = total - before

            s, t = returns[key]
            returns[key] = (s + was_seen[key], t + 1)

            s0, t0 = returns[(state, 0)]
            s1, t1 = returns[(state, 1)]
            val0 = safe_div(s0, t0)
            val1 = safe_div(s1, t1)

            pi[state] = 1 if val1 > val0 else 0

    Q = {k: safe_div(s, t) for (k, (s, t)) in returns.items()}

    return Q, pi


# todo: draw a surface
def draw_v(V):
    no_ace_player_sums = []
    no_ace_dealer_card = []
    no_ace_reward = []

    ace_player_sums = []
    ace_dealer_card = []
    ace_reward = []

    for k, v in V.iteritems():
        if k[2] is True:
            ace_player_sums.append(k[0])
            ace_dealer_card.append(k[1])
            ace_reward.append(v)
        else:
            no_ace_player_sums.append(k[0])
            no_ace_dealer_card.append(k[1])
            no_ace_reward.append(v)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(no_ace_dealer_card, no_ace_player_sums, no_ace_reward)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ace_dealer_card, ace_player_sums, ace_reward)
    plt.show()


def draw_optimal_policy(pi):
    x_1_with_ace = []
    y_1_with_ace = []

    x_0_with_ace = []
    y_0_with_ace = []

    x_1_without_ace = []
    y_1_without_ace = []

    x_0_without_ace = []
    y_0_without_ace = []

    for k, v in pi.iteritems():
        if k[2]:  # with unsable ace
            if v == 1:
                x_1_with_ace.append(k[1])
                y_1_with_ace.append(k[0])
            else:
                x_0_with_ace.append(k[1])
                y_0_with_ace.append(k[0])

        if not k[2]:  # without unsable ace
            if v == 1:
                x_1_without_ace.append(k[1])
                y_1_without_ace.append(k[0])
            else:
                x_0_without_ace.append(k[1])
                y_0_without_ace.append(k[0])

    trace1 = go.Scatter(
        x=x_1_with_ace,
        y=y_1_with_ace,
        mode='text',
        text=['hit'] * len(x_1_with_ace),
        textfont=dict(
            family='sans serif',
            size=10,
            color='#00FF00'
        )
    )

    trace2 = go.Scatter(
        x=x_0_with_ace,
        y=y_0_with_ace,
        mode='text',
        text=['stick'] * len(x_0_with_ace),
        textfont=dict(
            family='sans serif',
            size=10,
            color='#FF0000'
        )
    )

    data = [trace1, trace2]

    layout = go.Layout(
        title='Optimal policy with unsable ace',
        showlegend=False,
        xaxis=dict(
            title='dealer card'
        ),
        yaxis=dict(
            title='player sum'
        )
    )

    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig, filename='optimal-policy-with-ace')

    trace1 = go.Scatter(
        x=x_1_without_ace,
        y=y_1_without_ace,
        mode='text',
        text=['hit'] * len(x_1_without_ace),
        textfont=dict(
            family='sans serif',
            size=10,
            color='#00FF00'
        )
    )

    trace2 = go.Scatter(
        x=x_0_without_ace,
        y=y_0_without_ace,
        mode='text',
        text=['stick'] * len(x_0_without_ace),
        textfont=dict(
            family='sans serif',
            size=10,
            color='#FF0000'
        )
    )

    data = [trace1, trace2]

    layout = go.Layout(
        title='Optimal policy without unsable ace',
        showlegend=False,
        xaxis=dict(
            title='dealer card'
        ),
        yaxis=dict(
            title='player sum'
        )
    )

    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig, filename='optimal-policy-without-ace')


def main():
    V = first_visit_mc_prediction(500000)
    draw_v(V)

    Q, pi = mc_es(5000000)
    print "Optimal policy:", pi

    draw_optimal_policy(pi)


if __name__ == '__main__':
    main()
