import numpy

import numpy as np


def getStateValues(env):
    stateValues = np.zeros(env.observation_space.n)
    savedStateValues = np.copy(stateValues)
    finalQMatrix = np.zeros((env.observation_space.n, env.action_space.n))

    min_diff = 0.1e-10

    for i in range(10000):

        qMatrix = np.zeros((env.observation_space.n, env.action_space.n))

        for s in range(env.observation_space.n):
            for a in range(env.action_space.n):
                # print("Eval State:", s, "action", a)
                qMatrix[s][a] = probabilisticQValue(s, a, stateValues, env)

        # print("QM", qMatrix)

        for s in range(env.observation_space.n):
            stateValues[s] = np.max(qMatrix[s])

        diff = np.sum(np.fabs(stateValues - savedStateValues))

        finalQMatrix = qMatrix

        if diff < min_diff:
            break

        savedStateValues = np.copy(stateValues)
    else:
        print("WARN: reached end above min_level")

    stateActions = np.zeros(env.observation_space.n, numpy.int)

    for s in range(env.observation_space.n):
        stateActions[s] = np.argmax(finalQMatrix[s])

    return savedStateValues, stateActions

def probabilisticQValue(sourceS, action, savedStateValues, env):
    q = 0
    gamma = 0.9

    for prob, targetS, reward, final in env.P[sourceS][action]:
        q = q + prob * (reward + gamma * savedStateValues[targetS])

    return q

