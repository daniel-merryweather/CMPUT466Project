from tensorflow.python.ops.gen_math_ops import Max
import car
import numpy as np
import random
import pygame

import parameters as args

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, MaxPool1D, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error

import tensorflow as tf
import keras


from iteround import saferound

from collections import deque

class DQCar(car.Car):
	def __init__(self, x=0, y=0, r=90, w=15, h=30) -> None:
		super().__init__(x=0, y=0, r=90, w=15, h=30)
		self.reset(x,y,r,w,h)

		self.memory_size = 1000
		self.batch_size = 500

		self.actions = None
		self.lr = None
		self.gamma = None
		self.epsilon = None
		self.max_states = None
		self.q_table = None

		self.tm = None
		self.cm = None
		self.model = None

		self.discount_rate = 0.95

		self.optimizer = Adam(learning_rate=1e-2)
		self.loss_fn = mean_squared_error

		self.replay_memory = deque(maxlen=5000)

		self.input_shape = [4]
		self.n_outputs = 4 # Number of actions

		self.episode = 0
		self.reward = 0

		self.time_step = 0

		self.state = np.array([1.0, 1.0, 1.0, 0.0])

		self.max_vel = 50

		self.rand_generator = np.random.RandomState()

	def setup(self, actions, tm, cm, learning_rate = 0.9, reward_decay = 0.9, e_greedy = 0.9, max_states = 10000 ):
		self.actions = actions
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.max_states = max_states
		self.q_table = np.zeros((self.max_states, len(self.actions)), dtype = np.float64)

		self.tm = tm
		self.cm = cm

		self.sensorVals = np.ones(3)
		try:
			self.model = Sequential([
				Dense(8, activation="elu", input_shape=self.input_shape),
				Dense(6, activation="elu"),
				Dense(8, activation="elu"),
				Dense(self.n_outputs)
			])
		except Exception as e:
			print("-------\n",e,"\n------\n")

		self.model.summary()

	def sample_experiences(self, batch_size):
		indices = np.random.randint(len(self.replay_memory), size=batch_size)
		batch = [self.replay_memory[index] for index in indices]
		states, actions, rewards, next_states, dones = [
			np.array([experience[field_index] for experience in batch]) for field_index in range(5)
		]
		return states, actions, rewards, next_states, dones

	def play_one_step(self, state, epsilon):
		action = self.epsilon_greedy_policy(state, epsilon)
		next_state, reward, done = self.step(action)
		self.replay_memory.append((state, action, reward, next_state, done))
		return next_state, reward, done

	def training_step(self, batch_size):
		experiences = self.sample_experiences(batch_size)
		states, actions, rewards, next_states, dones = experiences
		next_Q_values = self.model.predict(next_states)
		max_next_Q_values = np.max(next_Q_values, axis=1)
		target_Q_values = (rewards +
						(1 - dones) * self.discount_rate * max_next_Q_values)
		target_Q_values = target_Q_values.reshape(-1, 1)
		mask = tf.one_hot(actions, self.n_outputs)
		with tf.GradientTape() as tape:
			all_Q_values = self.model(states)
			Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
			loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
		grads = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


	def argmax(self, q_values):
		top = float("-inf")
		ties = []

		for i in range(len(q_values)):
			if q_values[i] > top:
				top = q_values[i]
				ties = []

			if q_values[i] == top:
				ties.append(i)

		return self.rand_generator.choice(ties)


	def epsilon_greedy_policy(self, state, epsilon=0):
		if np.random.rand() < epsilon:
			action = np.random.randint(self.n_outputs)
			return action
		else:
			Q_values = self.model.predict(state[np.newaxis])
			action = self.argmax(Q_values[0])
			return action

	#def train(self, terminal_state, step):
	#	if len(self.replay_memory) < self.memory_size:
	#		return

	#	batch = random.sample(self.replay_memory, self.batch_size)

	#	current_states = np.array([transition[0]])


	def step(self, action):

		rotScalar = 0.1

		#print("Action: ", action, self.actions[action])

		if self.actions[action] == "a":
			self.rotate(rotScalar * self.vel[1] * 0.05)
		elif self.actions[action] == "d":
			self.rotate(-rotScalar * self.vel[1] * 0.05)
		elif self.actions[action] == "w":
			self.acc[1] = 10
		elif self.actions[action] == "s":
			self.acc[1] = -10
		elif action == "r": # Reset position
			self.reset()
		else:
			self.acc[1] = -3

		if self.collisionCheck(self.tm):
			self.reset(x=args.CAR_STARTING_POS[0], y=args.CAR_STARTING_POS[1])
			self.cm.currentcheckpoint=0
			self.reward = -1000
			print(self.episode, self.model.get_weights())

			if self.episode > 5:
				self.training_step(self.batch_size)
			self.episode += 1
			self.time_step = 0

			return np.array([1.0, 1.0, 1.0, 0.0]), -100, False


		temp_state = self.calculateSensorValues(self.tm)

		#print(temp_state)

		state = []

		for i in temp_state:
			state.append(round(i, 1))


		self.sensorVals = self.calculateSensorValues(self.tm)



		state.append(round(self.vel[1], 1))

		#print(state)

		if self.vel[1] < 1 and self.time_step > 100:
			return np.asarray(state), -5, False

		#if self.cm.checkCollision(self):
		#	return np.asarray(state), 50, False

		sensorCount = len(self.sensorVals)
		reward = 0

		rewardList = []

		#print(state)

		for i in range(sensorCount):

			reward += (1 - self.sensorVals[i]) * -10
			rewardList.append(reward)

		self.reward = reward
		return np.asarray(state), reward, False


	def handleInput(self, rotScalar=0.1, forwardSpeed=0.3):
		self.time_step += 1
		self.state, self.reward, done = self.play_one_step(self.state, 0.1)

		if self.time_step % 100 == 0:
			print(self.time_step)

		if self.time_step > 4000:
			print("hello")
			self.reset(x=args.CAR_STARTING_POS[0], y=args.CAR_STARTING_POS[1])
			self.time_step = 0
			self.episode += 1
			if self.episode > 5:
				self.training_step(self.batch_size)


	def handlePhysics(self, dt=0.01):
		if self.vel[1] < self.max_vel:
			self.vel = self.vel + self.acc * dt
		if self.vel[1] < 0:
			self.vel[1] = 0
		new_pos = self.pos + np.matmul(self.vel * dt, self.rotMat)

		if(new_pos[0] > 0 and new_pos[0] < args.WINDOW_SIZE[0] and new_pos[1] > 0 and new_pos[1] < args.WINDOW_SIZE[1]):
			self.pos = new_pos
