import pygame
import numpy as np

class Line:

	def __init__(self, x1, y1, x2, y2):
		self.update(x1, y1, x2, y2)

	def update(self, x1, y1, x2, y2):
		self.p1 = (x1, y1)
		self.p2 = (x2, y2)
		if(x2 < x1):
			temp = self.p1
			self.p1 = self.p2
			self.p2 = temp

		self.left = min(x1, x2)
		self.right = max(x1, x2)
		self.top = min(y1, y2)
		self.bottom = max(y1, y2)
		if(self.right-self.left == 0):
			self.slope = None
		else:
			self.slope = (self.bottom-self.top)/(self.right-self.left)
			if (self.p1[0] < self.p2[0] and self.p1[1] > self.p2[1] or
				self.p1[0] > self.p2[0] and self.p1[1] < self.p2[1]):
				self.slope *= -1

	def solveIntersection(self, other):
		if((self.slope == None and other.slope == None) or
			(self.slope != None and other.slope != None and abs(self.slope - other.slope) < 0.00001)):
			return None
		elif self.slope == None:
			x = self.p1[0]
			otherB = (other.p1[1]-other.p1[0]*other.slope)
			y = otherB + x * other.slope
			return (int(x),int(y))
		elif other.slope == None:
			x = other.p1[0]
			selfB = (self.p1[1]-self.p1[0]*self.slope)
			y = selfB + x * self.slope
			return (int(x),int(y))
		else:
			selfB = (self.p1[1]-self.p1[0]*self.slope)
			otherB = (other.p1[1]-other.p1[0]*other.slope)
			x = (otherB - selfB)/(self.slope - other.slope)
			y = selfB + x * self.slope
			return (int(x),int(y))

	def shift(self, xs, ys):
		self.update(self.p1[0] + xs, self.p1[1] + ys, self.p2[0] + xs, self.p2[1] + ys)

	def boundaryTest(self, other):
		return (self.left <= other.right and other.left <= self.right
			and self.top <= other.bottom and other.top <= self.bottom)

	def pointCollision(self, other, pos):
		left = max(self.left, other.left) - 1
		right = min(self.right, other.right) + 1
		top = max(self.top, other.top) - 1
		bottom = min(self.bottom, other.bottom) + 1
		return pos[0] >= left and pos[0] <= right and pos[1] >= top and pos[1] <= bottom

	def draw(self, window, debug=0):
		pygame.draw.line(window, (100,255,100), self.p1, self.p2)
		if debug > 0:
			pygame.draw.rect(window, (128,128,128),
				(self.left, self.top, self.right-self.left, self.bottom-self.top), 1)