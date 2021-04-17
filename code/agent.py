import pygame
import parameters as args
import numpy as np
from car import Car
from line import Line
from track import TrackSegment, TrackManager
from checkpoint import CheckpointManager
from network import NeuralNetwork
from copy import deepcopy

"""
Agent Class
	Connects car with agent behavior
"""
class Agent:
	def __init__(self, tm):
		self.car = Car(args.CAR_STARTING_POS[0],args.CAR_STARTING_POS[1])
		self.car.update()
		self.tm = tm
		self.cm = CheckpointManager()
		self.cm.generateCheckpoints(self.tm, n=30)

		self.terminated = False
		self.network = NeuralNetwork([len(self.car.sensorLines),3,2])
		
	def update(self):
		"""
		Update Function
			Updates car and neural network
		"""
		if self.terminated:
			return
		lastpos = deepcopy(self.car.pos)
		netVals = self.network.calculate(self.car.calculateSensorValues(self.tm))
		self.car.rotate((netVals[0]-0.5)*2 * self.car.vel[1] * 0.02)
		self.car.vel[1] = netVals[1]*200

		self.car.handlePhysics()
		self.car.update()
		self.cm.update(self.car)
		if self.car.collisionCheck(self.tm):
			self.terminated = True
		if (self.car.pos == lastpos).all():
			self.terminated = True

	def draw(self, window, color=(255,100,100)):
		"""
		Draw Function
			Draws agent and its checkpoint manager
		"""
		self.car.draw(window, color=color)
		self.cm.draw(window)


