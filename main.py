import gym
import numpy as np

import dpprob
import egreedy_lake

def translate(index: int):
    transvalues = ["LEFT", "DOWN", "RIGT", "-UP-"]
    return transvalues[index]

def printStateActions(stateActions, width, height):
    # translate = [0, 1, 2, 3]

    for w in range(height):
        for h in range(width):
            print(translate(stateActions[w * width + h]), end=" ")
        print()

# env = gym.make("FrozenLake8x8-v0", is_slippery=False)
# env = gym.make("FrozenLake-v0", is_slippery=False)
env = gym.make("CartPole-v1")

# stateActions = dpprob.getStateActions(env)
stateActions = egreedy_lake.getStateActions(env)

state = env.reset()
env.render()
dimension = int(np.sqrt(env.observation_space.n))
printStateActions(stateActions, dimension, dimension)

num_steps = 100
success = 0
attempts = 100

for tries in range(attempts):
    total = 0
    state = env.reset()
    next_step = stateActions[state]
    for sx in range(num_steps):
        new_state, reward, done, info = env.step(next_step)
        # env.render()

        next_step = stateActions[new_state]

        if done:
            if (reward > 0):
                success += 1
                # print("### success")
            else:
                noop = 0
                # print("### splash")
            break
    # print("Attempt:", tries, "- reward:", total)
print("Success:", success, "/", attempts)


