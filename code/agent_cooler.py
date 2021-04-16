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
	def __init__(self, actions, car, tm, learning_rate = 0.9, reward_decay = 0.9, e_greedy = 0.9, max_states = 10000):
		self.actions = actions
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.max_states = max_states
		self.max_states = 10 * (8 ** 7)
		self.q_table = np.zeros((self.max_states, len(self.actions)), dtype = np.float32)
		#self.q_table = np.random.rand(self.max_states, len(self.actions))
		#self.q_table = np.random.randint(10, size=(self.max_states, len(self.actions)))
		self.last_index = sensors_to_index(car, tm)
		self.car = car
		self.tm = tm
		self.last_action = "w"
		self.rewards = 0

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

	def choose_action(self):
		index = sensors_to_index(self.car, self.tm)
		#if self.last_index == index:
			#return self.last_action
		
		if np.random.uniform() < self.epsilon:
			state_action = np.argmax(self.q_table[index])
			action = self.actions[state_action]
		else:
			action = np.random.choice(self.actions)
		self.next_action = action
		return self.last_action;

	def learn(self, reward):
		index = sensors_to_index(self.car, self.tm)
		#if self.last_index == index:
			#return
		sensors = self.car.calculateSensorValues(self.tm)
		for sensor in sensors:
			reward += sensor * (300/7)
		self.rewards += reward
		
		last_action_id = self.actions.index(self.last_action)
		self.last_action = self.next_action
		
		q_predict = self.q_table[self.last_index, last_action_id]
		if np.max(self.q_table[self.last_index]) == 0:
			q_target = reward
		else:
			q_target = reward + self.gamma * np.max(self.q_table[index])
		self.q_table[self.last_index, last_action_id] += self.lr * (q_target - q_predict)
		self.last_index = index
		return

def sensors_to_index(car, tm):
	vel = car.vel[1]
	sensors = car.calculateSensorValues(tm)

	vel = vel//20
	if vel >= 10:
		vel = 9

	index = 0
	for i in range(len(sensors)):
		sensors[i] = int(sensors[i] * 8)
		if sensors[i] >= 8:
			sensors[i] = 7
		index += sensors[i] * (8 ** i)

	index += vel * (8 ** 7)
	return int(index)
