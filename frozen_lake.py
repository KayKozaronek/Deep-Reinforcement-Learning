import gym 
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0")

n_games = 1000
win_rate = []
scores = []

for episode in range(n_games):
    done = False
    observation = env.reset()
    score = 0

    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        score += reward
    
    scores.append(score)

# Win percentage / 10 games over time
    if episode % 10 == 0:
        average = np.mean(scores[-10:])
        win_rate.append(average)

# Plot win % over trailing 10 games 
plt.plot(win_rate)
plt.show()
