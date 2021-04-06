import pygame
import parameters as args
import numpy as np
from line import Line

class Car:
	def __init__(self, x=0, y=0, r=90, w=15, h=30):
		self.reset(x,y,r,w,h)

	def reset(self,x=0, y=0, r=90, w=15, h=30):
		self.pos = np.array([x,y])
		self.w = w
		self.h = h
		self.r = r

		self.acc = np.array([0,0])
		self.vel = np.array([0,0])

		self.setRotationMatrix()

	def rotate(self, theta):
		self.r += theta
		self.setRotationMatrix()

	def setRotationMatrix(self):
		r = np.radians(self.r)
		self.rotMat = np.array(((np.cos(r), -np.sin(r)), (np.sin(r), np.cos(r))))

	def generateBody(self):
		bodyPoints = np.array([
			[-self.w/2,-self.h/2], [self.w/2,-self.h/2],
			[self.w/2,self.h/2], [-self.w/2,self.h/2]])

		bodyOffset = np.array([0,self.h/4])
		bodyPoints += bodyOffset

		bodyPoints = np.matmul(bodyPoints, self.rotMat)

		self.carBody = bodyPoints + self.pos
		return self.carBody

	def generateSensorLines(self, n=7, rotStep=15, maxLength=120):
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
		for i in range(len(self.carBody)):
			bodyLine = Line(self.carBody[i][0], self.carBody[i][1],
							self.carBody[(i+1)%4][0], self.carBody[(i+1)%4][1])
			if tm.checkLineIntersection(bodyLine):
				return True
		return False


	def handlePhysics(self, dt=0.01):
		self.vel = self.vel + self.acc * dt
		if self.vel[1] < 0:
			self.vel[1] = 0
		new_pos = self.pos + np.matmul(self.vel * dt, self.rotMat)

		if(new_pos[0] > 0 and new_pos[0] < args.WINDOW_SIZE[0] and new_pos[1] > 0 and new_pos[1] < args.WINDOW_SIZE[1]):
			self.pos = new_pos

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

	def handleAgentInput(self, command, rotScalar=0.1, forwardSpeed=0.3):
                # commands expect format ex. "w", "wa", "a"
		
		if "a" in command:
			self.rotate(rotScalar * self.vel[1] * 0.2)
		if "d" in command:
			self.rotate(-rotScalar * self.vel[1] * 0.2)
		if "w" in command:
			self.acc[1] = 10
		elif "s" in command:
			self.acc[1] = -10
		elif "r" in command: # Reset position
			self.reset()
		else:
			self.acc[1] = -3

	def update(self):
		self.generateBody()
		self.generateSensorLines()

	def draw(self, window):
		pygame.draw.polygon(window, (255,100,100), self.carBody)
		for l in self.sensorLines:
			pygame.draw.line(window, (255,255,100), self.pos, l)
