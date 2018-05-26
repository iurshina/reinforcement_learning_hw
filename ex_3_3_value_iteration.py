

class Transition:
    def __init__(self, action, destination, reward):
        self.action = action
        self.destination = destination
        self.reward = reward


def value_iteration(mdp, theta, gamma):
    V = dict([(s, 0) for s in mdp.keys()])

    while True:
        dlt = 0

        for s in mdp.keys():
            v = V[s]
            max_r = 0
            for t in mdp[s]:
                r = t.reward + gamma * V[t.destination]
                if max_r < r:
                    max_r = r

            V[s] = max_r
            dlt = max(dlt, abs(v - V[s]))

        if dlt < theta:
            break

    return V


def best_policy(mdp, V, gamma):
    pi = {}
    for s in mdp.keys():
        actions = [t.action for t in mdp[s]]
        max_utility = 0
        a_with_max_u = 0
        for a in actions:
            u = utility(a, s, mdp, V, gamma)
            if u > max_utility:
                max_utility = u
                a_with_max_u = a

        pi[s] = a_with_max_u

    return pi


def utility(a, s, mdp, V, gamma):
    transitions = mdp[s]
    for t in transitions:
        if t.action == a:
            return t.reward + gamma * V[t.destination]


def main():
    theta = 0.000000001
    gamma = 0.2
    mdp = {0: [Transition(-1, 0, 0), Transition(1, 1, 0)],
           1: [Transition(-1, 0, 1), Transition(1, 2, 0)],
           2: [Transition(-1, 1, 0), Transition(1, 3, 0)],
           3: [Transition(-1, 2, 0), Transition(1, 4, 0)],
           4: [Transition(-1, 3, 0), Transition(1, 5, 5)],
           5: [Transition(-1, 4, 0), Transition(1, 5, 0)]}

    V = value_iteration(mdp, theta, gamma)

    print "gamma:", gamma, "optimal policy:", best_policy(mdp, V, gamma)


if __name__ == '__main__':
    main()