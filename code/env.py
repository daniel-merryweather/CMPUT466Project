import utils
import pygame

import parameters as args

class Env:

	def __init__(self, window, font, clock):

		self.actions = ['w', 'a', 's', 'd']

		self.window = window
		self.font = font
		self.clock = clock

		self.tm = utils.setupTrack() 	  # Track Manager
		self.cm = utils.setupCheckPoints(self.tm) # CheckPoint Manager
		self.agent = utils.setAgent(self.actions)

		#self.agents = [self.agent,
		#			   self.agent,
		#			   self.agent,
		#			   self.agent]

		#self.agents[0].color = (255,100,100)
		#self.agents[1].color = (100,255,100)
		#self.agents[2].color = (100,100,255)
		#self.agents[3].color = (255,255,255)


		try:
			#for agent in self.agents:
			self.agent.setup(self.actions, self.tm, self.cm)
			self.agent.update()
		except Exception as e:
			print(e)

	def run(self):

		best_weights = None
		best_time = 0

		running = True
		while running:

			#for e in pygame.event.get():
			#	if e.type == pygame.QUIT:
			#		running = False

			# Draw track
			#utils.drawFrame(self.window, self.tm)

			#for agent in self.agents:

			for e in pygame.event.get():
				if e.type == pygame.QUIT:
					running = False

			utils.drawFrame(self.window, self.tm)
			self.agent.handleInput()

			deltaTime = 0.01
			if self.clock.get_fps() > 0:
				deltaTime = 1/self.clock.get_fps()

			self.agent.handlePhysics(dt=deltaTime)
			self.agent.update()
			self.agent.draw(self.window)

			self.cm.update(self.agent)
			self.cm.draw(self.window)

			sensorVals = self.agent.calculateSensorValues(self.tm)
			sensorCount = len(sensorVals)

			for i in range(sensorCount):

				barWidth = 10
				barHeight = 100
				pygame.draw.rect(self.window, (255,255,100),
					(args.WINDOW_SIZE[0]-2*barWidth*sensorCount-10+barWidth + i*2*barWidth, args.WINDOW_SIZE[1]-10-sensorVals[i]*barHeight,
					barWidth, sensorVals[i]*barHeight), 0)

			utils.drawDisplay(self.font, self.window, self.clock, self.agent)

			if self.agent.best_time > best_time:
			
				best_time = self.agent.best_time
				best_weights = self.agent.best_weights
				print("-------\nNEW BEST\n------\n")
				print(best_time, best_weights)

			if self.agent.time_step == 0 and best_weights is not None:
				print("-------\nUpdate\n--------\n")
				print(best_time, best_weights)
				self.agent.model.set_weights(best_weights)
				self.agent.model.best_time = best_time


		pygame.quit()