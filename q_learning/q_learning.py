# Q learning environment
import matplotlib.pyplot as plt 
import numpy as np 
import gym 
from agent import Agent

def update_q_value(reward): # I have to write this function
    

# Q is a dictionary with states and actions as keys


# Initialize Q for all states s and actions a

Q = INIT # HOW DO I DO THAT?
alpha = 0.001
gamma = 0.9
e_max = 1
e_min = 0.01
EPISODES = 5000

scores = []
avg_score = []

env = gym.make("CartPole-v1")


for episode in range(EPISODES):
    # Initialize state s
    # IS state env or observation?
    # I think the observation is the state
    state = env.reset()
    done = False

    # For each step of the episode:
    while done == False:
        #env.render()

        # Choose an action a in current state s with an epsilon-greedy strategy
        action = agent.choose_action(state)

        # Perform action a, get new state s' and get reward r
        new_state, reward, done, info = env.step(action)

        # Plug in reward r into update equation to get new estimation for Q(s,a)
        q_value = update_q_value(reward) # I have to write this function

        # Set the old state s to the new state s'
        state = new_state
    
    # Decrement epsilon over time to some minimal value
    epsilon = epsilon * 0.95

    scores.append(reward)

    if episode % 100 == 0:
        avg_score.append(np.mean(scores[-100:]))

# Plot average score over 100 games (learning curve)
plt.plot(avg_score)
plt.show()
