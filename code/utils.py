
import parameters as args
from agent_cooler import QLearningTable
from track import TrackSegment, TrackManager
from checkpoint import CheckpointManager
from car import Car
from qcar import QCar
from dqcar2 import DQCar



import pygame
import sys

def setupTrack():
	"""
	This method creates the segments of the track with a specified curvature.
	"""
	trackSegments = [
		TrackSegment((200,100),(900,100), curveMagnitude=0),
		TrackSegment((900,100),(1100,300), curveMagnitude=-7),
		TrackSegment((1100,300),(900,500), curveMagnitude=-7),
		TrackSegment((900,500),(800,400), curveMagnitude=-3.5),
		TrackSegment((800,400),(600,250), curveMagnitude=5),
		TrackSegment((600,250),(400,400), curveMagnitude=5),
		TrackSegment((400,400),(500,500), curveMagnitude=3.5),
		TrackSegment((500,500),(600,600), curveMagnitude=-3.5),
		TrackSegment((600,600),(500,700), curveMagnitude=-3.5),
		TrackSegment((500,700),(200,700), curveMagnitude=0),
		TrackSegment((200,700),(100,600), curveMagnitude=-3.5),
		TrackSegment((100,600),(200,500), curveMagnitude=-3.5),
		TrackSegment((200,500),(300,400), curveMagnitude=3.5),
		TrackSegment((300,400),(200,300), curveMagnitude=3.5),
		TrackSegment((200,300),(100,200), curveMagnitude=-3.5),
		TrackSegment((100,200),(200,100), curveMagnitude=-3.5)
		]

	tm = TrackManager()
	tm.addSegments(trackSegments)
	tm.gatherTrackPoints()
	tm.generateLines()
	tm.close()
	return tm

def setupCheckPoints(trackManager):
	"""
	This method creates the checkpoints of the track.
	"""
	cm = CheckpointManager()
	cm.generateCheckpoints(trackManager)
	return cm


def setAgent(actions):
	"""
	This method allows for setting other agents as input parameters at execution.
	"""
	try:
		# Potential use of deep q-learning agent
		if sys.argv[-1] == "dq":
			return DQCar(args.CAR_STARTING_POS[0],args.CAR_STARTING_POS[1])
		# Implement Q-Learning Agent
		elif sys.argv[-1] == "ql":
			return QCar(args.CAR_STARTING_POS[0],args.CAR_STARTING_POS[1])

	# Default is a drivable car
	except:
		return Car(args.CAR_STARTING_POS[0],args.CAR_STARTING_POS[1])

def drawFrame(window, tm):
	"""
	This method draws the track and window.
	"""

	window.fill((80,80,80))

	for i in range(int(args.WINDOW_SIZE[0]/100)+1):
		pygame.draw.line(window, (120,120,120), (i*100,0), (i*100,args.WINDOW_SIZE[1]))
	for i in range(int(args.WINDOW_SIZE[1]/100)+1):
		pygame.draw.line(window, (120,120,120), (0,i*100), (args.WINDOW_SIZE[0], i*100))

	tm.draw(window, debug=1)

def drawDisplay(font, window, clock, car):
	fps = font.render("FPS: " + str(int(clock.get_fps())), True, (255,255,100))
	disc = font.render("DISCLAIMER: Current version does not represent final product", True, (255,255,100))
	speed = font.render("Speed (units/second): " + str(car.vel[1].round(1)), True, (255,255,100))
	accel = font.render("Acceleration (units^2/second): " + str(car.acc[1]), True, (255,255,100))
	reward = font.render("Reward: " + str(car.reward), True, (255,255,100))
	window.blit(fps, (10,10))
	window.blit(disc, (10,args.WINDOW_SIZE[1]-25))
	window.blit(speed, (args.WINDOW_SIZE[0]-500,args.WINDOW_SIZE[1]-110))
	window.blit(accel, (args.WINDOW_SIZE[0]-500,args.WINDOW_SIZE[1]-60))
	window.blit(reward, (args.WINDOW_SIZE[0]-500,args.WINDOW_SIZE[1]-20))

	pygame.display.flip()
	clock.tick(0)