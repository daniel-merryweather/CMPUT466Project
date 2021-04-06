import car
import numpy as np
import pygame


class QCar(car.Car):
	def __init__(self, x=0, y=0, r=90, w=15, h=30) -> None:
		super().__init__()
		self.reset(x,y,r,w,h)

		self.actions = None
		self.lr = None
		self.gamma = None
		self.epsilon = None
		self.max_states = None
		self.q_table = None
		self.reward = 0

		self.last_state = None
		self.last_action = None
		self.last_reward = None

	def setup(self, actions, learning_rate = 0.9, reward_decay = 0.9, e_greedy = 0.9, max_states = 10000):
		self.actions = actions
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.max_states = max_states
		self.q_table = np.zeros((self.max_states, len(self.actions)), dtype = np.float64)

	def handleInput(self, rotScalar=0.1, forwardSpeed=0.3):
		keys = pygame.key.get_pressed()
		if keys[pygame.K_a]:
			self.rotate(rotScalar * self.vel[1] * 0.05)
		if keys[pygame.K_d]:
			self.rotate(-rotScalar * self.vel[1] * 0.05)
		if keys[pygame.K_w]:
			self.acc[1] = 10
		elif keys[pygame.K_s]:
			self.acc[1] = -10
		elif keys[pygame.K_r]: # Reset position
			self.reset()
		else:
			self.acc[1] = -3