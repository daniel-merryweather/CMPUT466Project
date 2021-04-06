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
		self.bestnet = None
		for i in range(n):
			self.agents.append(Agent(tm))

	def update(self):
		generationEnded = True
		for a in self.agents:
			a.update()
			if a.terminated == False:
				generationEnded = False
		return generationEnded

	def evolve(self):
		agentnetpool = []
		bestcheckpoints = -1
		bestnet = None
		for a in self.agents:
			if a.cm.currentcheckpoint < bestcheckpoints:
				bestcheckpoints = a.cm.currentcheckpoint
				bestnet = a.network
			for i in range(a.cm.currentcheckpoint**2):
				agentnetpool.append(a.network)
		self.bestnet = bestnet

		if len(agentnetpool) == 0:
			self.agents = []
			for i in range(self.n):
				self.agents.append(Agent(self.tm))
			return
		newagents = []
		for i in range(self.n):
			parent_a = random.choice(agentnetpool)
			parent_b = random.choice(agentnetpool)
			newnet = parent_a.cross(parent_b)
			newagent = Agent(self.tm)
			newagent.network = newnet
			newagents.append(newagent)
		self.agents = newagents

	def saveBestNetwork(self):
		if self.bestnet != None:
			self.bestnet.save("saved-best")

	def draw(self, window):
		for a in self.agents:
			a.draw(window)
