import pygame
import parameters as args
from car import Car
import numpy as np

class Agent:
	def __init__(self):
		self.car = Car(100,100)
        self.actions = np.array([[0,0,0,0]]) # W A S D 
    #def step(self, action):   
    '''
    Do the action on the car, there is 2^4 possible actions.
    Return
        observation (object) = position of the car, the current velcoties, accleration, rotation
        reward (float) = -1 per step + k (if position = next checkpoint)
    '''
        