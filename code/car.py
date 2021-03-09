import pygame
import parameters as args
import numpy as np

class Car:
	def __init__(self, x=0, y=0, r=0, w=25, h=45):
		self.reset(x,y,r,w,h)

	def reset(self,x=0, y=0, r=0, w=25, h=45):
		self.pos = np.array([x,y])
		self.w = w
		self.h = h
		self.r = r

		self.acc = np.array([0,0])
		self.vel = np.array([0,0])

		self.rotMat = np.array([[1,0],[0,1]])

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

		return bodyPoints + self.pos

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

		return lines + self.pos

	def handlePhysics(self, dt=0.01):
		self.vel = self.vel + self.acc * dt
		if self.vel[1] < 0:
			self.vel[1] = 0
		new_pos = self.pos + np.matmul(self.vel * dt, self.rotMat)

		if(new_pos[0] > 0 and new_pos[0] < args.WINDOW_SIZE[0] and new_pos[1] > 0 and new_pos[1] < args.WINDOW_SIZE[1]):
			self.pos = new_pos

	def handleInput(self, rotScalar=0.2, forwardSpeed=0.3):
		keys = pygame.key.get_pressed()
		if keys[pygame.K_a]:
			self.rotate(rotScalar * self.vel[1] * 0.05)
		if keys[pygame.K_d]:
			self.rotate(-rotScalar * self.vel[1] * 0.05)
		if keys[pygame.K_w]:
			self.acc[1] = 1
		elif keys[pygame.K_s]:
			self.acc[1] = -2
		elif keys[pygame.K_r]: # Reset position
			self.reset()
		else:
			self.acc[1] = -1

	def draw(self, window):
		carBody = self.generateBody()
		sensorLines = self.generateSensorLines()

		pygame.draw.polygon(window, (255,100,100), carBody)
		for l in sensorLines:
			pygame.draw.line(window, (100,100,255), self.pos, l)