# Q learning environment
import matplotlib.pyplot as plt 
import numpy as np 
import gym 
from agent import Agent

if __name__ == '__main__':

    # Variables
    learning_rate = 0.001
    discount_factor = 0.9
    epsilon = 1.0
    eps_min = 0.01
    eps_dec = 0.999995
    n_actions = 4
    n_states = 16
    EPISODES = 500000

    # Initialization of environment & Agent
    env = gym.make("FrozenLake-v0")
    agent = Agent(learning_rate, discount_factor, n_actions, n_states, epsilon, eps_min, eps_dec)

    # Keep track of scores & average win percentage
    scores = []
    avg_score = []


    for episode in range(EPISODES):
        # Initialize state s
        state = env.reset()
        done = False
        score = 0

        # For each step of the episode:
        while not done:
            # Choose an action a in current state s with an epsilon-greedy strategy
            action = agent.choose_action(state)

            # Perform action a, get new state s' and get reward r
            new_state, reward, done, info = env.step(action)

            # Plug in reward r into update equation to get new estimation for Q(s,a)
            agent.learn(state, action, reward, new_state)

            # Update score
            score += reward 

            # Set the old state s to the new state s'
            state = new_state
        
        scores.append(score)

        if episode % 100 == 0:
            win_pct = np.mean(scores[-100:])
            avg_score.append(win_pct)
            if episode % 1000 == 0:
                print("The win percentage after episode ", episode, " is ", win_pct )
                print("Epsilon is currently ", round(agent.epsilon, 2))

    # Plot average score over 100 games (learning curve)
    plt.plot(avg_score)
    plt.show()
