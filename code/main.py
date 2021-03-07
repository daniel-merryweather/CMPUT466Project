import pygame
from car import Car

def init():
	pygame.init()

def loop():
	running = True

	window = pygame.display.set_mode([720,480])

	car = Car(100,100)

	while(running):
		for e in pygame.event.get():
			if e.type == pygame.QUIT:
				running = False

		window.fill((230,230,230))

		car.handleInput()
		car.handlePhysics()
		car.draw(window)

		pygame.display.flip()

	pygame.quit()

def main():
	init()
	loop()

if __name__ == '__main__':
	main()