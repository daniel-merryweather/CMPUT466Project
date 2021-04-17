import pygame
import numpy as np
from line import Line

class CheckpointManager:

	def __init__(self):
		self.currentcheckpoint = 0
		self.checkpoints = []

	def generateCheckpoints(self, tm, n=30, offset=10):
		size = len(tm.iPoints)
		self.n = n
		self.checkpoints = []
		for i in range(n):
			dn = int((size-offset)/n)
			self.checkpoints.append((tm.iPoints[offset+dn*i], tm.oPoints[offset+dn*i]))

	def checkLineIntersection(self, line):
		current = self.checkpoints[self.currentcheckpoint%len(self.checkpoints)]
		l = Line(current[0][0], current[0][1], current[1][0], current[1][1])
		pos = line.solveIntersection(l)
		if pos != None and line.pointCollision(l, pos):
			return True
		return False

	def checkCollision(self, car):
		for i in range(len(car.carBody)):
			bodyLine = Line(car.carBody[i][0], car.carBody[i][1],
							car.carBody[(i+1)%4][0], car.carBody[(i+1)%4][1])
			if self.checkLineIntersection(bodyLine):
				return True
		return False

	def update(self, car):
		if self.checkCollision(car):
			self.currentcheckpoint += 1
			return 1
		return 0

	def draw(self, window, debug=0):
		for i in range(len(self.checkpoints)):
			c = self.checkpoints[i]
			if i == self.currentcheckpoint%len(self.checkpoints):
				pygame.draw.line(window, (255,100,255), c[0], c[1])
			elif debug >= 1:
				pygame.draw.line(window, (100,255,255), c[0], c[1])
