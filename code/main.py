import pygame
import parameters as args
import numpy as np
from line import Line
from track import TrackSegment, TrackManager
from checkpoint import CheckpointManager
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
	
	# Initial Track Settings
	convoluted_track = True # Set to true to try agent on a basic circular track
	if (convoluted_track):
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
			TrackSegment((100,200),(200,100), curveMagnitude=-3.5)]
	else:
		trackSegments = [
			TrackSegment((300,100),(700,100), curveMagnitude=0),
			TrackSegment((700,100),(900,300), curveMagnitude=-7),
			TrackSegment((900,300),(700,500), curveMagnitude=-7),
			TrackSegment((700,500),(300,500), curveMagnitude=0),
			TrackSegment((300,500),(100,300), curveMagnitude=-7),
			TrackSegment((100,300),(300,100), curveMagnitude=-7)]

	# Track Manager setup
	tm = TrackManager()
	tm.addSegments(trackSegments)
	tm.gatherTrackPoints()
	tm.generateLines()
	tm.close()
	
	# Agent Manager setup
	am = AgentManager(tm, n=10)

	# Updating Graphics and handling input
	while(running):
		# Input Handling
		for e in pygame.event.get():
			if e.type == pygame.QUIT:
				running = False
			if e.type == pygame.KEYDOWN:
				if e.key == pygame.K_e:
					am.evolve()
				if e.key == pygame.K_RIGHT:
					for i in range(200):
						am.update()

		# Background rendering
		window.fill((80,80,80))

		for i in range(int(args.WINDOW_SIZE[0]/100)+1):
			pygame.draw.line(window, (120,120,120), (i*100,0), (i*100,args.WINDOW_SIZE[1]))
		for i in range(int(args.WINDOW_SIZE[1]/100)+1):
			pygame.draw.line(window, (120,120,120), (0,i*100), (args.WINDOW_SIZE[0], i*100))

		# Render track
		tm.draw(window)
		
		# Keep clock for FPS counter
		deltaTime = 0.01
		if clock.get_fps() > 0:
			deltaTime = 1/clock.get_fps()

		# When a generation completely dies automatically evolve
		if am.update():
			am.evolve()
		am.draw(window)

		# Display information
		fps = font.render("FPS: " + str(int(clock.get_fps())), True, (255,255,100))
		description = "Generation #" + str(am.generationNumber) + ", Living Agents: " + str(am.getLivingAgentCount()) + ", If generation gets stuck press E to manually evolve."
		desc = font.render(description, True, (255,150,0))
		
		window.blit(fps, (10,10))
		window.blit(desc, (10,args.WINDOW_SIZE[1]-25))

		pygame.display.flip()
		clock.tick(0)
	pygame.quit()

def main():
	init()
	loop()

if __name__ == '__main__':
	main()
