import pygame
import sys

import parameters as args
from env import Env

def main():

	# Initialize Pygame
	pygame.init()
	# Set pygame settings
	window = pygame.display.set_mode(args.WINDOW_SIZE)
	font = pygame.font.Font(None, 30)
	clock = pygame.time.Clock()
	# Initialize Environment
	env = Env(window, font, clock)
	env.run()
	# Exit Pygame
	pygame.quit()

main()