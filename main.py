import gym
import dpprob
import egreedy


# env = gym.make("FrozenLake8x8-v0")
env = gym.make("FrozenLake-v0")

# stateActions = dpprob.getStateActions(env)
stateActions = egreedy.getStateActions(env)

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
            total = reward
            # print("Reward:", reward)
            break
    if (total>0):
        success = success + 1
    # print("Attempt:", tries, "- reward:", total)
print("Success:", success, "/", attempts)