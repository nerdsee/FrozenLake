import numpy as np


def getStateActions(env):
    training_rate = 0.9
    gamma = 0.95
    episodes = 10000
    reduction = 0.95
    epsilon = 0.5
    epsmin = 0.1

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    oldQ = Q.copy()

    for ep in range(episodes):
        state = env.reset()
        steps = 0

        actions = []
        visited_states = [state]

        if epsilon > epsmin:
            epsilon = epsilon * reduction

        while True:
            steps += 1
            reward = 0
            action = getNextAction(state, Q, epsilon)

            if state == 0 and action == 0:
                pass

            actions.append(action)
            new_state, ret_reward, done, info = env.step(action)

            if done:
                if ret_reward == 0:
                    reward = 0
                else:
                    reward = 10
            else:
                if new_state not in visited_states:
                    reward = 1
                    visited_states.append(new_state)
                else:
                    reward = -1

            currentQ = Q[state][action]
            maxQList = Q[new_state].copy()
            maxQ = np.amax(maxQList)
            newQ = currentQ + training_rate * (reward + gamma * maxQ - currentQ)
            Q[state][action] = newQ

            state = new_state

            if done:
                if reward > 0:
                    # print("done after", steps, "steps", "reward", reward, "state", state)
                    pass
                # print(actions)
                break

        diff = np.sum(np.fabs(Q - oldQ))
        oldQ = Q.copy()

        if (ep % 1000 == 0):
            print("Diff after {} rounds: {}".format(ep, diff))

    print("Q:", Q)

    stateActions = np.zeros(env.observation_space.n, np.int)

    for s in range(env.observation_space.n):
        stateActions[s] = np.argmax(Q[s])

    return stateActions


def getNextAction(state, Q, epsilon):
    action = 0

    rand_num = np.random.random_sample()

    if (rand_num < epsilon):
        # explore
        action = np.random.randint(0, Q.shape[1])
    else:
        # exploit
        # max = np.argmax(Q[state])
        maxq = np.max(Q[state])
        options = np.flatnonzero(Q[state] == maxq)
        numopt = options.shape[0]
        sel = np.random.randint(0, numopt)
        action = options[sel]

    return action
