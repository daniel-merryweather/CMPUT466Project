import pygame
import parameters as args
import numpy as np
from line import Line
from track import TrackSegment, TrackManager
from checkpoint import CheckpointManager
#from agent import Agent
from agentmanager import AgentManager

# Initialization
def init():
	pygame.init()

# Main Continious loop
def loop():
	running = True

	window = pygame.display.set_mode(args.WINDOW_SIZE)
	font = pygame.font.Font(None, 30)
	clock = pygame.time.Clock()


	#car = Car(args.CAR_STARTING_POS[0],args.CAR_STARTING_POS[1])
	
	# Initial Track Settings
	trackSegments = [
		TrackSegment((200,100),(500,100), curveMagnitude=0),
		TrackSegment((500,100),(900,100), curveMagnitude=0),
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
		TrackSegment((100,200),(200,100), curveMagnitude=-3.5)]

	tm = TrackManager()
	tm.addSegments(trackSegments)
	tm.gatherTrackPoints()
	tm.generateLines()
	tm.close()
	
	#agent1 = Agent(tm)
	am = AgentManager(tm, n=10)

	# Updating Graphics and handling input
	while(running):
		for e in pygame.event.get():
			if e.type == pygame.QUIT:
				running = False
			if e.type == pygame.KEYDOWN:
				if e.key == pygame.K_e:
					am.evolve()
				if e.key == pygame.K_RIGHT:
					for i in range(200):
						am.update()
				if e.key == pygame.K_s:
					am.saveBestNetwork()

		window.fill((80,80,80))

		for i in range(int(args.WINDOW_SIZE[0]/100)+1):
			pygame.draw.line(window, (120,120,120), (i*100,0), (i*100,args.WINDOW_SIZE[1]))
		for i in range(int(args.WINDOW_SIZE[1]/100)+1):
			pygame.draw.line(window, (120,120,120), (0,i*100), (args.WINDOW_SIZE[0], i*100))

		tm.draw(window, debug=1)
		
		deltaTime = 0.01
		if clock.get_fps() > 0:
			deltaTime = 1/clock.get_fps()

		am.update()
		am.draw(window)

		# Display sensor readings as bar graph
		# sensorVals = car.calculateSensorValues(tm)
		# sensorCount = len(sensorVals)
		# for i in range(sensorCount):
		# 	barWidth = 10
		# 	barHeight = 100
		# 	pygame.draw.rect(window, (255,255,100),
		# 		(args.WINDOW_SIZE[0]-2*barWidth*sensorCount-10+barWidth + i*2*barWidth, args.WINDOW_SIZE[1]-10-sensorVals[i]*barHeight,
		# 		barWidth, sensorVals[i]*barHeight), 0)

		fps = font.render("FPS: " + str(int(clock.get_fps())), True, (255,255,100))
		disc = font.render("DISCLAIMER: Current version does not represent final product", True, (255,255,100))
		#speed = font.render("Speed (units/second): " + str(car.vel[1].round(1)), True, (255,255,100))
		#accel = font.render("Acceleration (units^2/second): " + str(car.acc[1]), True, (255,255,100))
		window.blit(fps, (10,10))
		window.blit(disc, (10,args.WINDOW_SIZE[1]-25))
		#window.blit(speed, (args.WINDOW_SIZE[0]-500,args.WINDOW_SIZE[1]-110))
		#window.blit(accel, (args.WINDOW_SIZE[0]-500,args.WINDOW_SIZE[1]-60))

		pygame.display.flip()
		clock.tick(0)
	pygame.quit()

def main():
	init()
	loop()

if __name__ == '__main__':
	main()
