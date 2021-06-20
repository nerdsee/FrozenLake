import numpy as np


def getStateActions(env):
    training_rate = 0.9
    gamma = 0.95
    episodes = 100000

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    oldQ = Q.copy()

    for ep in range(episodes):
        state = env.reset()
        steps = 0

        while True:
            steps += 1
            action = getNextAction(state, Q)

            new_state, reward, done, info = env.step(action)

            # reward = reward * 10 + steps

            currentQ = Q[state][action]
            maxQList = Q[new_state].copy()
            maxQ = np.amax(maxQList)
            newQ = currentQ + training_rate * (reward + gamma * maxQ - currentQ)
            Q[state][action] = newQ
            state = new_state

            if done:
                # print("done after", steps, "steps", "reward", reward)
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


def getNextAction(state, Q):
    epsilon = 0.7
    action = 0

    rand_num = np.random.random_sample()

    if (rand_num < epsilon):
        # explore
        action = np.random.randint(0, Q.shape[1])
    else:
        # exploit
        action = np.argmax(Q[state])

    return action
