import pygame
import parameters as args
import numpy as np
from car import Car
from line import Line
from track import TrackSegment, TrackManager
from checkpoint import CheckpointManager
from network import NeuralNetwork
from copy import deepcopy


'''
state space:
	[sv1, sv2,..., svn]			sv -> sensor value
action space:
	[coast,left,right]
'''

class Agent:
	def __init__(self, tm):
		self.car = Car(args.CAR_STARTING_POS[0],args.CAR_STARTING_POS[1])
		self.car.update()
		self.tm = tm
		self.cm = CheckpointManager()
		self.cm.generateCheckpoints(self.tm, n=30)

		self.terminated = False

		#print([len(self.car.sensorLines),3,3,2])
		self.network = NeuralNetwork([len(self.car.sensorLines),3,2])
		# self.car.vel[1] = 100
		# self.action = 0
		
	def update(self):
		if self.terminated:
			return
		lastpos = deepcopy(self.car.pos)
		#self.takeAction(self.action)
		netVals = self.network.calculate(self.car.calculateSensorValues(self.tm))
		self.car.rotate((netVals[0]-0.5)*2 * self.car.vel[1] * 0.02) #
		#if netVals[1] < 0.1:
		#	netVals[1] = 0.1
		self.car.vel[1] = netVals[1]*200

		self.car.handlePhysics()
		self.car.update()
		self.cm.update(self.car)
		if self.car.collisionCheck(self.tm):
			#self.car.reset(x=args.CAR_STARTING_POS[0], y=args.CAR_STARTING_POS[1])
			#self.cm.currentcheckpoint = 0
			self.terminated = True
		if (self.car.pos == lastpos).all():
			self.terminated = True
			# self.car.vel[1] = 100
		# if self.car.pos[0] > 500:
			# self.action = 1

	# def takeAction(self, a, rotScalar=0.02):
	# 	if a == 1:
	# 		self.car.rotate(self.car.vel[1]*rotScalar)
	# 	elif a == 2:
	# 		self.car.rotate(-self.car.vel[1]*rotScalar)

	def draw(self, window, color=(255,100,100)):
		self.car.draw(window, color=color)
		self.cm.draw(window)


