# Based on A Begginners Guide to Q-Learning by Chathurangi Shyalika
	# https://towardsdatascience.com/a-beginners-guide-to-q-learning-c3e2a30a653c
# And code based on ChathurangiShyalika/Agent_Path_Following
	#https://github.com/ChathurangiShyalika/Agent_Path_Following/blob/master/agent.py

import pygame
import car
import parameters as args
import numpy as np
from line import Line

class QLearningTable:
	def __init__(self, actions, learning_rate = 0.9, reward_decay = 0.9, e_greedy = 0.9, max_states = 10000):
		self.actions = actions
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.max_states = max_states
		self.max_states = 10 * (4 ** 7)
		self.q_table = np.zeros((self.max_states, len(self.actions)), dtype = np.float64)

	'''
	def choose_action(self, curr_state):
		if curr_state >= self.max_states:
			return 'r';
		elif np.random.uniform() < self.epsilon:
			state_action = np.argmax(self.q_table[curr_state])
			action = self.actions[state_action]
		else:
			action = np.random.choice(self.actions)
		return action;

	def learn(self, curr_state, action_id, reward):
		if (curr_state + 1) >= self.max_states:
			return
		q_predict = self.q_table[curr_state, action_id]
		if np.max(self.q_table[curr_state]) == 0:
			q_target = reward
		else:
			q_target = reward + self.gamma * np.max(self.q_table[curr_state + 1])
		self.q_table[curr_state, action_id] += q_predict + self.lr * (q_target - q_predict)
		return
	'''

	def choose_action(self, car, tm):
		index = sensors_to_index(car, tm)
		
		if np.random.uniform() < self.epsilon:
			state_action = np.argmax(self.q_table[index])
			action = self.actions[state_action]
		else:
			action = np.random.choice(self.actions)
		return action;

	def learn(self, car, tm, action_id, reward):
		index = sensors_to_index(car, tm)
		
		q_predict = self.q_table[index, action_id]
		if np.max(self.q_table[index]) == 0:
			q_target = reward
		else:
			q_target = reward + self.gamma * np.max(self.q_table[index])
		self.q_table[index, action_id] += q_predict + self.lr * (q_target - q_predict)
		return

def sensors_to_index(car, tm):
	vel = car.vel[1]
	sensors = car.calculateSensorValues(tm)

	vel = vel//10
	if vel >= 10:
		vel = 9

	index = 0
	for i in range(len(sensors)):
		sensors[i] = int(sensors[i] * 4)
		if sensors[i] >= 4:
			sensors[i] = 3
		index += sensors[i] * (3 ** i)

	index += vel * (4 ** 7)
	return int(index)
