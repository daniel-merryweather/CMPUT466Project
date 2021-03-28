# Referenced https://www.geeksforgeeks.org/q-learning-in-python/

import pygame
import car
import parameters as args
import numpy as np
from line import Line

class Agent:
	def __init__(self):
        self.state = np.ones(num_actions, dtype = float) # The sensor readings
        self.actions = np.zeros(8) # One-hot-vector with actions: left (l), right (r), forward (f), back (b), lf, rf, lb, rb
        self.Q = defaultdict(lambda: np.zeros(len(self.actions))) # The Q-function, which takes a state, and action, and returns expected value

    def createEpsilonGreedyPolicy(self, epsilon):
        def policyFunction(state):
            actions_probability = np.ones(num_actions, dtype = float) * epsilon / len(self.actions)
            greedy_action = np.argmax(self.Q[self.state])
            actions_probability[greedy_action] += (1.0 - epsilon)
            return actions_probability
        return policyFunction

    def learnPolicy(self, num_episodes, discount_factor = 1.0,
							alpha = 0.6, epsilon = 0.1):
        """
        Q-Learning algorithm: Off-policy TD control.
        Finds the optimal greedy policy while improving
        following an epsilon-greedy policy"""
        

        # Keeps track of useful statistics
        stats = plotting.EpisodeStats(
            episode_lengths = np.zeros(num_episodes),
            episode_rewards = np.zeros(num_episodes))	
        
        # Create an epsilon greedy policy function
        # appropriately for environment action space
        policy = createEpsilonGreedyPolicy(epsilon)
        
        # For every episode
        for ith_episode in range(num_episodes):
            
            # Reset the environment and pick the first action
            # Reset the car position/world
            state = env.reset()

            # might need to use itertools? not really sure what that is
            for t in itertools.count():
                
                # get probabilities of all actions from current state
                action_probabilities = policy(state)

                # choose action according to
                # the probability distribution
                action = np.random.choice(np.arange(
                        len(action_probabilities)),
                        p = action_probabilities)

                # take action and get reward, transit to next state
                # evn.step would be moving the car
                next_state, reward, done, _ = env.step(action)

                # Update statistics
                stats.episode_rewards[ith_episode] += reward
                stats.episode_lengths[ith_episode] = t
                
                # TD Update
                best_next_action = np.argmax(Q[next_state])	
                td_target = reward + discount_factor * self.Q[next_state][best_next_action]
                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += alpha * td_delta

                # done is True if episode terminated
                if done:
                    break
                    
                state = next_state
        
        return Q, stats
