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
		"""
		This method will create 2 Dense layers for a neural network with an output vector correlating to Q-values for certain actions.
		"""
		#optimizer = SGD(learning_rate=0.001)
		optimizer = Adam(learning_rate=0.001)
		#loss_fn = mean_squared_error
		loss_fn = Huber()

		# Initialize Sequential NN
		model = Sequential()
		model.add(Dense(24, activation="elu", input_shape=input_shape))
		model.add(Dense(12, activation="elu"))
		model.add(Dense(output_shape, activation='linear'))
		model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

		return model



	def setup(self, actions, tm, cm):
		"""
		This method will initialize the Deep Q Car model and target model with the track manager and checkpoint manager with the appropriate actions.
		"""
		self.actions = actions
		self.tm = tm	# track manager
		self.cm = cm	# checkpoint manager

		self.model = self.create_model(self.input_shape, len(self.actions))
		self.target_model = self.create_model(self.input_shape, len(self.actions))
		self.target_model.set_weights(self.model.get_weights())


	def handleInput(self, rotScalar=0.1, forwardSpeed=0.3):
		"""
		This method will handle the car input method that controls the actions handling the car
		"""
		self.time_step += 1
		self.target_steps += 1
		# Perform one step for the car
		next_state, reward, done = self.play_one_step(self.curr_state)

		# Update current state and total reward
		self.curr_state = next_state
		self.reward += reward

		# Every 4 steps train the model using the replay buffer
		if self.target_steps % 4 == 0:
			self.train()
		# If the car hits the wall and lasts more than 500 steps it will train the model and set the target model with the trained weights
		elif done and self.target_steps >= 500:
			self.train()
			self.target_model.set_weights(self.model.get_weights())
			self.target_steps = 0
			self.time_step = 0
			self.reset(x=args.CAR_STARTING_POS[0], y=args.CAR_STARTING_POS[1])

		# If the model hits the wall without making it to the minimum step number reset the position without training and setting the weights
		if done == True:
			self.reward = 0
			self.time_step = 0
			self.target_steps = 0
			self.reset(x=args.CAR_STARTING_POS[0], y=args.CAR_STARTING_POS[1])


	def play_one_step(self, state):
		"""
		This method will perform one step transition for the car evaluting the action to be taken using either the model to predict an action to be taken or taking a random action if exploration is selected.
		"""

		# If the random number is under the epsilon threshold, pick a random action
		if np.random.rand() < self.epsilon:
			action = np.random.randint(len(self.actions))
		# Else pick the highest q-valued action
		else:
			Q_values = self.model.predict(state[np.newaxis])
			action = np.argmax(Q_values[0])

		# Take an environment step
		next_state, reward, done = self.step(action)

		# Append this transition to the replay memory
		self.replay_memory.append([state, action, reward, next_state, done])

		return next_state, reward, done

	def train(self):
		"""
		This method will handle training the model with the transitions observed in the replay memory over the previous time steps.
		"""
		# If the replay memory does not contain enough data to sample
		if len(self.replay_memory) < self.batch_size:
			return
		# Sample a mini batch of data from the replay memory
		mini_batch = random.sample(self.replay_memory, self.batch_size)
		# Get the list of current states sample
		curr_states = np.array([transition[0] for transition in mini_batch])
		# Predict the Q-values for the states sampled
		curr_q_list = self.model.predict(curr_states)
		# Get the list of next states sampled
		new_curr_states = np.array([transition[3] for transition in mini_batch])
		# Predict the Q-values for the next states sampled
		future_q_list = self.target_model.predict(new_curr_states)

		# Set a list of states to fit the model with
		states = []
		# Set a list of q-values to fit the model against as output
		qs = []

		# Iterate through the mini batch performing updates on the action-value estimate
		for index, (state, action, reward, next_state, done) in enumerate(mini_batch):
			# If the state is not a terminal state, perform a full update
			if not done:
				max_future_q = reward + self.gamma * np.max(future_q_list[index])
			# If the state is the terminal perform a final update adding only the reward
			else:
				max_future_q = reward
			# Get the current Q-value for the current batch index
			curr_q = curr_q_list[index]
			# Perform the update for the action-value estimate
			curr_q[action] = curr_q[action] + self.learning_rate * (max_future_q - curr_q[action])

			# Append the state to the list of states to fit
			states.append(state)
			# Append the action-value estimate to the list of q-values to fit the model against
			qs.append(curr_q)

		# Fit the model to the states and state-action values observed
		self.model.fit(np.array(states), np.array(qs), batch_size=self.batch_size, shuffle=True, verbose=0)

	def step(self, action):
		"""
		This method will perform an environment interaction by moving the car based on the action taken by the model
		"""

		rotScalar = 0.2

		reward = 0
		# Perform an action on the car based on which action is selected
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

		# If the car collides with the wall, reset the car position and give a negative reward
		if self.collisionCheck(self.tm):
			self.reset(x=args.CAR_STARTING_POS[0], y=args.CAR_STARTING_POS[1])
			self.cm.currentcheckpoint=0

			return np.zeros(self.input_shape[0]), -100, True


		# Get the state of the sensors on the car
		temp_state = self.calculateSensorValues(self.tm)
		# Initialize a state list
		state = []
		# Append each state after rounding the value to 2 decimal places to simplify the state space
		for i in temp_state:
			state.append(1 - round(i, self.rounding))


		# The following is a potential negative reward system for training the agent based on higher sensor readings and not only collisions with the wall.

		"""
		self.sensorVals = self.calculateSensorValues(self.tm)

		sensorCount = len(self.sensorVals)

		rewardList = []

		for i in range(sensorCount):

			reward += -(((1 - self.sensorVals[i]) * -1) ** 2)
			rewardList.append(reward)
		"""

		# If the car does not hit a wall return a 0 reward
		reward = 0
		return np.asarray(state), reward, False

	def handlePhysics(self, dt=0.01):
		"""
		This method is overwritten from the car class to modify velocity values and acceleration to prevent the car from not moving at all.
		"""
		if self.vel[1] < self.max_vel:
			self.vel = self.vel + self.acc * dt
		if self.vel[1] < self.min_vel:
			self.vel[1] = self.min_vel
		new_pos = self.pos + np.matmul(self.vel * dt, self.rotMat)

		if(new_pos[0] > 0 and new_pos[0] < args.WINDOW_SIZE[0] and new_pos[1] > 0 and new_pos[1] < args.WINDOW_SIZE[1]):
			self.pos = new_pos

	def draw(self, window):
		"""
		This method is overwritten from the car class to allow a modified car color in the case of multiple instances being trained.
		"""
		pygame.draw.polygon(window, self.color, self.carBody)
		for l in self.sensorLines:
			pygame.draw.line(window, (255,255,100), self.pos, l)
