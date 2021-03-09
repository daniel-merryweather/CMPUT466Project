import pygame
import parameters as args
import numpy as np

class MapGenerator:
	def __init__(self):

'''
Has to dnyamically create the walls over the 2D course. The walls should stop the car. (Possible negative reward aswell)
Possibly using a random number generator that picks between left center right.
Where left will create walls that go slightly to the left etc
'''