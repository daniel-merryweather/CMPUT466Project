# Based on A Begginners Guide to Q-Learning by Chathurangi Shyalika
	# https://towardsdatascience.com/a-beginners-guide-to-q-learning-c3e2a30a653c
# And code based on ChathurangiShyalika/Agent_Path_Following
	#https://github.com/ChathurangiShyalika/Agent_Path_Following/blob/master/agent.py

import pygame
import car
import parameters as args
import numpy as np
import pickle
from line import Line

class QLearningTable:
	def __init__(self, actions, learning_rate = 0.9, reward_decay = 0.9, e_greedy = 1, max_states = 25):
		self.actions = actions
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.max_states = max_states
		self.q_table = np.zeros((self.max_states, len(self.actions)), dtype = np.float64)
		self.previous_index = 0

		try:
			with open ('q_data.pkl', 'rb') as f:
				self.q_table = pickle.load(f)
		except:
			print("No previous pickle file")

	def save_output(self):
		with open ('q_data.pkl', "wb") as f:
			pickle.dump(self.q_table,f)
			print("Saved pickle file")

	def choose_action(self, car, tm):
		index = sensors_to_index(car, tm)
		
		if np.random.uniform() < self.epsilon:
			state_action = np.argmax(self.q_table[index])
			action = self.actions[state_action]
		else:
			action = np.random.choice(self.actions)

		self.previous_index = sensors_to_index(car, tm)
		return action

	def learn(self, car, tm, action_id, reward):
		new_Index = sensors_to_index(car, tm)
		
		q_predict = self.q_table[self.previous_index, action_id]
		if np.max(self.q_table[self.previous_index]) == 0:
			q_target = reward
		else:
			q_target = reward + self.gamma * np.max(self.q_table[new_Index])
		self.q_table[self.previous_index, action_id] += self.lr * (q_target - q_predict)
		print("q[%d][%d] = %f , vel = %f" % (self.previous_index, action_id, self.q_table[self.previous_index, action_id], car.vel[1]))
		return

# Just using the left and right most sensor readings for the index
def sensors_to_index(car, tm):
	sensors = car.calculateSensorValues(tm)
	index = int( (sensors[0] - sensors[6]) * car.vel[1]/10)
	return index
