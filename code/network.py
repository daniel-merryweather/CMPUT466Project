import numpy as np
import random as rand
from copy import deepcopy
import gc
from math import *
import sys
import pickle

"""
Neural Network Class
"""
class NeuralNetwork():

    def __init__(self, nodeSizes):
        """
        Initialization method
            Creates a neural network given an array of integers representing nodes in each layer
        """
        self.nodeSizes = nodeSizes

        self.weights = [None] * (len(nodeSizes)-1)
        self.biases = np.random.rand(len(nodeSizes)-1)
        self.biases *= 2
        self.biases -= 1
        self.biases /= 3
        for i in range(1,len(nodeSizes)):
            self.weights[i-1] = np.random.rand(nodeSizes[i-1],nodeSizes[i])
            self.weights[i-1] *= 2
            self.weights[i-1] -= 1
        
    def calculate(self, inputLayer):
        """
        Feed Forward Method
            Calculates the output of the neural network given an array of input values
        """
        for i in range(0,len(self.weights)-1):
            inputLayer = np.tanh(np.dot(inputLayer,self.weights[i]) + self.biases[i])
        inputLayer = np.dot(inputLayer,self.weights[len(self.weights)-1])
        inputLayer = np.clip(inputLayer, -1,1)
        return inputLayer

    def cross(self, other, epsilon=0.05):
        """
        Cross Method
            Splices two different NeuralNetwork's into one
        """
        adjustmentScalar = 0.2
        newNetwork = NeuralNetwork(self.nodeSizes)
        for li in range(len(self.weights)):
            for i in range(len(self.weights[li])):
                r = rand.uniform(0,1)
                if r < epsilon:
                    newNetwork.weights[li][i] += rand.uniform(-adjustmentScalar,adjustmentScalar)
                elif r < 0.5-epsilon:
                    newNetwork.weights[li][i] = other.weights[li][i]
                else:
                    newNetwork.weights[li][i] = self.weights[li][i]

        for bi in range(len(self.biases)):
            r = rand.uniform(0,1)
            if(r < epsilon):
                newNetwork.biases[bi] += rand.uniform(-adjustmentScalar,adjustmentScalar)
            elif(r < 0.5-epsilon):
                newNetwork.biases[bi] = other.biases[bi]
            else:
                newNetwork.biases[bi] = self.biases[bi]
        return newNetwork

    def addVariance(self, adjustmentScalar=0.02):
        """
        Add Variance Method
            Adds slight variations to the neural network
        """
        for li in range(len(self.weights)):
            for i in range(len(self.weights[li])):
                self.weights[li][i] += rand.uniform(-adjustmentScalar,adjustmentScalar)

        for bi in range(len(self.biases)):
            self.biases[bi] += rand.uniform(-adjustmentScalar,adjustmentScalar)

    def isEqualTo(self, other):
        """
        Neural Network Comparison Function
        """
        for li in range(len(self.weights)):
            for i in range(len(self.weights[li])):
                for wi in range(len(self.weights[li][i])):
                    if self.weights[li][i][wi] != other.weights[li][i][wi]:
                        return False

        for bi in range(len(self.biases)):
            if self.biases[bi] != other.biases[bi]:
                return False
        return True
    
