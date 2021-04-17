from track import TrackSegment, TrackManager
from checkpoint import CheckpointManager
from agent import Agent
import parameters as args
import numpy as np
import random

class AgentManager:
	def __init__(self, tm, n=1):
		self.tm = tm
		self.agents = []
		self.n = n
		self.generationNumber = 0
		for i in range(n):
			self.agents.append(Agent(tm))

	def update(self):
		bestCheckpoints = -1
		self.bestAgent = -1
		generationEnded = True
		for i in range(len(self.agents)):
			a = self.agents[i]
			a.update()
			if a.cm.currentcheckpoint > bestCheckpoints:
				bestCheckpoints = a.cm.currentcheckpoint
				self.bestAgent = i
			if a.terminated == False:
				generationEnded = False

		return generationEnded

	def evolve(self):
		agentnetpool = []
		for a in self.agents:
			for i in range(a.cm.currentcheckpoint**2):
				agentnetpool.append(a.network)

		if len(agentnetpool) == 0:
			self.agents = []
			for i in range(self.n):
				self.agents.append(Agent(self.tm))
			return
		else:
			self.generationNumber += 1
		newagents = []
		for i in range(self.n):
			parent_a = random.choice(agentnetpool)
			parent_b = random.choice(agentnetpool)
			newnet = parent_a.cross(parent_b)
			newagent = Agent(self.tm)
			newagent.network = newnet
			newagents.append(newagent)

		self.agents = []
		for na in newagents:
			for a in self.agents:
				if na.network.isEqualTo(a.network):
					na.network.addVariance()
					break
			self.agents.append(na)
		print("New Generation of " + str(len(self.agents)) + " Agents.")

	def getLivingAgentCount(self):
		count = 0
		for a in self.agents:
			if a.terminated == False:
				count += 1
		return count

	def saveBestNetwork(self):
		if self.bestnet != None:
			self.bestnet.save("saved-best")

	def draw(self, window):
		for i in range(len(self.agents)):
			if i == self.bestAgent:
				self.agents[i].draw(window, color=(255,150,0))
			else:
				self.agents[i].draw(window)
