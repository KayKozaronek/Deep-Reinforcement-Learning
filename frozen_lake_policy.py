import gym 
import numpy as np 
import matplotlib.pyplot as plt

# Maze Outline 
# SFFF
# FHFH
# FFFH
# HFFG

# Moves 
# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3

# key = states, value, action
policy = {
    0:1, 1:2, 2:1, 3:0,
    4:1, 6:1,
    8:2, 9:1, 10:1,
    13:2, 14:2}

env = gym.make("FrozenLake-v0")
n_games= 1000
win_pct = []
scores = []

for episode in range(n_games):
    score = 0
    done = False
    observation = env.reset()

    while not done:
        # take an action
        action = policy[observation]
        observation, reward, done, info = env.step(action)
        score += reward

    scores.append(score)

    if episode % 10 == 0:
        avg = np.mean(scores[-10:])
        win_pct.append(avg)

plt.plot(win_pct)
plt.show()