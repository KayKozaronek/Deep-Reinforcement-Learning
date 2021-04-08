import numpy as np

class Agent():
    def __init__(self, learning_rate, discount_factor, n_actions, 
                 n_states, epsilon, eps_min, eps_dec):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec

        self.Q = {}

        self.init_Q()
    
    def init_Q(self):
        self.Q = {(state, action): 0.0 for action in range(self.n_actions) \
                    for state in range(self.n_states)}

        # for state in range(self.n_states):
        #     for action in range(self.n_actions):
        #         self.Q[(state, action)] = 0.0
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            # random action
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            # find optimal action
            actions = np.array([self.Q[(state, a)] \
                                for a in range(self.n_actions)])
            action = np.argmax(actions) # Choose alternative for tie braking
            #action = np.random.choice(np.flatnonzero(b == b.max()))  
        return action
    
    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon * self.eps_dec 
        else:
            self.epsilon = self.eps_min

    def learn(self, state, action, reward, new_state):
        actions = np.array([self.Q[(state, a) for a in range(n_actions)]])
        a_max = np.argmax(actions)

        self.Q[(state, action)] += self.lr * (reward + self.gamma *
                                            self.Q[(new_state, a_max)] - 
                                            self.Q[(state, action)])

        self.decrement_epsilon()