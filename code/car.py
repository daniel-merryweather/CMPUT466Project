import pygame
import parameters as args
import numpy as np
from line import Line

"""
Car Class
	Handles car position, rendering and sensors
"""
class Car:
	def __init__(self, x=0, y=0, r=90, w=15, h=30):
		self.reset(x,y,r,w,h)

	def reset(self,x=0, y=0, r=90, w=15, h=30):
		"""
		Reset Function
			Resets the car to the starting position, acceleration and velocity
		"""
		self.pos = np.array([x,y])
		self.w = w
		self.h = h
		self.r = r

		self.acc = np.array([0,0])
		self.vel = np.array([0,0])

		self.setRotationMatrix()

	def rotate(self, theta):
		"""
		Rotate Function
			Rotates the car in degrees
		"""
		self.r += theta
		self.setRotationMatrix()

	def setRotationMatrix(self):
		"""
		Set Rotation Matrix Function
			Calculates the rotation matrix for efficient calculation
		"""
		r = np.radians(self.r)
		self.rotMat = np.array(((np.cos(r), -np.sin(r)), (np.sin(r), np.cos(r))))

	def generateBody(self):
		"""
		Generate Body Function
			Calculates body car verticies
		"""
		bodyPoints = np.array([
			[-self.w/2,-self.h/2], [self.w/2,-self.h/2],
			[self.w/2,self.h/2], [-self.w/2,self.h/2]])

		bodyOffset = np.array([0,self.h/4])
		bodyPoints += bodyOffset

		bodyPoints = np.matmul(bodyPoints, self.rotMat)

		self.carBody = bodyPoints + self.pos
		return self.carBody

	def generateSensorLines(self, n=7, rotStep=20, maxLength=150):
		"""
		Generate Sensor Lines Function
			Dynamically generates sensor lines for the car
		"""
		lines = []
		for i in range(n):
			index = i-np.floor(n/2)
			r = np.radians(rotStep * index)
			line = np.array([0,1 - np.abs(index)/n])
			lRotMat = np.array(((0, 0), (np.sin(r), np.cos(r))))
			line = np.matmul(line, lRotMat)
			line *= maxLength
			lines.append(line)

		lines = np.array(lines)
		lines = np.matmul(lines, self.rotMat)

		self.sensorLines = lines + self.pos
		return lines + self.pos

	def calculateSensorValues(self, tm):
		"""
		Calculate Sensor Values Function
			Calculates the values of intersection for each sensor and the track
		"""
		values = [1]*len(self.sensorLines)
		for i in range(len(self.sensorLines)):
			s = self.sensorLines[i]
			sensor = Line(self.pos[0], self.pos[1], s[0], s[1])
			positions = tm.lineIntersectionPosition(sensor)
			for p in positions:
				sensorLength = np.sqrt(np.sum((self.pos - s)**2))
				sensorDepth = np.sqrt(np.sum((self.pos - p)**2))
				val = sensorDepth/sensorLength
				if val < values[i]:
					values[i] = val
		return values

	def collisionCheck(self, tm):
		"""
		Collision Check Function
			Checks if the car has crashed into the track
		"""
		for i in range(len(self.carBody)):
			bodyLine = Line(self.carBody[i][0], self.carBody[i][1],
							self.carBody[(i+1)%4][0], self.carBody[(i+1)%4][1])
			if tm.checkLineIntersection(bodyLine):
				return True
		return False


	def handlePhysics(self, dt=0.01):
		"""
		Handle Physics Function
			Handles car physics updates
		"""
		self.vel = self.vel + self.acc * dt
		if self.vel[1] < 0:
			self.vel[1] = 0
		new_pos = self.pos + np.matmul(self.vel * dt, self.rotMat)

		if(new_pos[0] > 0 and new_pos[0] < args.WINDOW_SIZE[0] and new_pos[1] > 0 and new_pos[1] < args.WINDOW_SIZE[1]):
			self.pos = new_pos

	def update(self):
		"""
		Update Function
			General function to generate body and lines (needed for change in position)
		"""
		self.generateBody()
		self.generateSensorLines()

	def draw(self, window, color=(255,100,100)):
		"""
		Draw Function
			Draws the car
		"""
		pygame.draw.polygon(window, color, self.carBody)
		for l in self.sensorLines:
			pygame.draw.line(window, (255,255,100), self.pos, l)
