from tensorflow.python.ops.gen_math_ops import Max
import car
import numpy as np
import random
import pygame

import parameters as args

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, MaxPool1D, Conv1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mean_squared_error, Huber

import tensorflow as tf
import keras


from iteround import saferound

from collections import deque

class DQCar(car.Car):
	"""
	Implementation is based on the generic Deep Q-Learning algorithm
	https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb
	"""

	def __init__(self, x=0, y=0, r=90, w=15, h=30) -> None:
		super().__init__(x=0, y=0, r=90, w=15, h=30)
		self.reset(x,y,r,w,h)

		# Managers
		self.tm = None
		self.cm = None

		# Models
		self.model = None
		self.target_model = None

		# Q-table values
		self.actions = None		# w, a, s, d
		self.max_states = None	# The maximum number of states possible
		self.input_shape = [3]	# Number of input parameters to the NN

		# Q-variables
		self.gamma = 0.75
		self.epsilon = 0.05
		self.learning_rate = 0.9

		# Memory Buffer
		self.replay_memory = deque(maxlen=50000)
		self.batch_size = 128

		# Sensors on the car
		self.sensorVals = np.ones(self.input_shape[0])
		# Random number generator
		self.rand_generator = np.random.RandomState()

		# Number of decimal places for sensor readings
		self.rounding = 2

		# Environment data
		self.best_time = 0
		self.time_step = 0
		self.episode = 0
		self.reward = 0

		# Current State
		self.curr_state = np.zeros(self.input_shape[0])

		# Speed limits
		self.max_vel = 100
		self.min_vel = 30

		# Vehicle Color
		self.color = (255,100,100)

		self.target_steps = 0



	def create_model(self, input_shape, output_shape):

		#optimizer = SGD(learning_rate=0.001)
		optimizer = Adam(learning_rate=0.001)
		#loss_fn = mean_squared_error
		loss_fn = Huber()

		model = Sequential()
		model.add(Dense(24, activation="elu", input_shape=input_shape))
		model.add(Dense(12, activation="elu"))
		model.add(Dense(output_shape, activation='linear'))
		model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

		return model



	def setup(self, actions, tm, cm):

		self.actions = actions
		self.tm = tm
		self.cm = cm

		self.model = self.create_model(self.input_shape, len(self.actions))
		self.target_model = self.create_model(self.input_shape, len(self.actions))
		self.target_model.set_weights(self.model.get_weights())


	def handleInput(self, rotScalar=0.1, forwardSpeed=0.3):
		self.time_step += 1
		self.target_steps += 1
		next_state, reward, done = self.play_one_step(self.curr_state)

		# Update current state and total reward
		self.curr_state = next_state
		self.reward += reward

		if self.target_steps % 4 == 0:
			self.train()
		elif done and self.target_steps >= 100:
			self.train()
			self.target_model.set_weights(self.model.get_weights())
			self.target_steps = 0
			self.time_step = 0
			self.reset(x=args.CAR_STARTING_POS[0], y=args.CAR_STARTING_POS[1])

		if done == True:
			self.reward = 0
			self.time_step = 0
			self.target_steps = 0
			self.reset(x=args.CAR_STARTING_POS[0], y=args.CAR_STARTING_POS[1])


	def play_one_step(self, state):

		# Perform Epsilon Greedy exploration
		if np.random.rand() < self.epsilon:
			action = np.random.randint(len(self.actions))
		else:
			Q_values = self.model.predict(state[np.newaxis])
			action = np.argmax(Q_values[0])
			print("Action:", action, "State:", state, "Q:", Q_values)

		# Take an environment step
		next_state, reward, done = self.step(action)

		# Append this transition to the replay memory
		self.replay_memory.append([state, action, reward, next_state, done])

		return next_state, reward, done

	def train(self):

		if len(self.replay_memory) < self.batch_size:
			return

		mini_batch = random.sample(self.replay_memory, self.batch_size)
		curr_states = np.array([transition[0] for transition in mini_batch])
		curr_q_list = self.model.predict(curr_states)
		new_curr_states = np.array([transition[3] for transition in mini_batch])
		future_q_list = self.target_model.predict(new_curr_states)

		states = []
		qs = []

		for index, (state, action, reward, next_state, done) in enumerate(mini_batch):

			if not done:
				max_future_q = reward + self.gamma * np.max(future_q_list[index])
			else:
				max_future_q = reward

			curr_q = curr_q_list[index]
			curr_q[action] = curr_q[action] + self.learning_rate * (max_future_q - curr_q[action])


			states.append(state)
			qs.append(curr_q)

		self.model.fit(np.array(states), np.array(qs), batch_size=self.batch_size, shuffle=True)

	def step(self, action):

		rotScalar = 0.2

		reward = 0

		if self.actions[action] == "a":
			self.rotate(rotScalar * self.vel[1] * 0.05)
		elif self.actions[action] == "d":
			self.rotate(-rotScalar * self.vel[1] * 0.05)
		elif self.actions[action] == "w":
			self.acc[1] = 10
		elif self.actions[action] == "s":
			self.acc[1] = -10
			reward = -self.time_step
		elif action == "r": # Reset position
			self.reset()
		else:
			self.acc[1] = -3

		if self.collisionCheck(self.tm):
			self.reset(x=args.CAR_STARTING_POS[0], y=args.CAR_STARTING_POS[1])
			self.cm.currentcheckpoint=0

			return np.zeros(self.input_shape[0]), -100, True


		temp_state = self.calculateSensorValues(self.tm)

		state = []

		for i in temp_state:
			state.append(1 - round(i, self.rounding))


		self.sensorVals = self.calculateSensorValues(self.tm)

		sensorCount = len(self.sensorVals)

		rewardList = []

		for i in range(sensorCount):

			reward += -(((1 - self.sensorVals[i]) * -1) ** 2)
			rewardList.append(reward)

		reward = 0
		return np.asarray(state), reward, False

	def handlePhysics(self, dt=0.01):
		if self.vel[1] < self.max_vel:
			self.vel = self.vel + self.acc * dt
		if self.vel[1] < self.min_vel:
			self.vel[1] = self.min_vel
		new_pos = self.pos + np.matmul(self.vel * dt, self.rotMat)

		if(new_pos[0] > 0 and new_pos[0] < args.WINDOW_SIZE[0] and new_pos[1] > 0 and new_pos[1] < args.WINDOW_SIZE[1]):
			self.pos = new_pos


	def draw(self, window):
		pygame.draw.polygon(window, self.color, self.carBody)
		for l in self.sensorLines:
			pygame.draw.line(window, (255,255,100), self.pos, l)
