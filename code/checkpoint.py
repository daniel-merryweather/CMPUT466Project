import pygame
import numpy as np
from line import Line

"""
Checkpoint Manager Class
	Handles all the checkpoints and keeps track of the current checkpoint
"""
class CheckpointManager:

	def __init__(self):
		self.currentcheckpoint = 0
		self.checkpoints = []

	def generateCheckpoints(self, tm, n=30, offset=10):
		"""
		Generate Checkpoints Function
			Dynamically generates the checkpoints from track data
		"""
		size = len(tm.iPoints)
		self.n = n
		self.checkpoints = []
		for i in range(n):
			dn = int((size-offset)/n)
			self.checkpoints.append((tm.iPoints[offset+dn*i], tm.oPoints[offset+dn*i]))

	def checkLineIntersection(self, line):
		"""
		Check Line Intersection Function
			Given any line, checks intersection with the current checkpoint
		"""
		current = self.checkpoints[self.currentcheckpoint%len(self.checkpoints)]
		l = Line(current[0][0], current[0][1], current[1][0], current[1][1])
		pos = line.solveIntersection(l)
		if pos != None and line.pointCollision(l, pos):
			return True
		return False

	def checkCollision(self, car):
		"""
		Check Collision Function
			Checks if the given car has hit its current goal checkpoint
		"""
		for i in range(len(car.carBody)):
			bodyLine = Line(car.carBody[i][0], car.carBody[i][1],
							car.carBody[(i+1)%4][0], car.carBody[(i+1)%4][1])
			if self.checkLineIntersection(bodyLine):
				return True
		return False

	def update(self, car):
		"""
		Update Function
			Updates the current checkpoint if reached
		"""
		if self.checkCollision(car):
			self.currentcheckpoint += 1
			return 1
		return 0

	def draw(self, window, debug=0):
		"""
		Draw Function
			Draws the goal checkpoint
			If debug >= 1 will also draw all other checkpoints
		"""
		for i in range(len(self.checkpoints)):
			c = self.checkpoints[i]
			if i == self.currentcheckpoint%len(self.checkpoints):
				pygame.draw.line(window, (255,100,255), c[0], c[1])
			elif debug >= 1:
				pygame.draw.line(window, (100,255,255), c[0], c[1])
