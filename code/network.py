import numpy as np
import random

class MultiplicativeNetwork:
	def __init__(self, structure):
		self.layers = []
		for i in range(len(structure)-1):
			self.layers.append(Layer(structure[i+1],structure[i]))

	def calculate(self, inputVals):
		nextVals = []
		for li in range(len(self.layers)):
			nextVals = self.layers[li].calculate(inputVals)
			inputVals = nextVals
		return nextVals

class Layer:
	def __init__(self, length, input_n):
		self.length = length
		self.nodes = []
		for i in range(length):
			self.nodes.append(Node(input_n))

	def calculate(self, inputVals):
		output = []
		for i in range(self.length):
			output.append(self.nodes[i].calculate(inputVals))
		return output

class Node:
	def __init__(self, input_n):
		self.w = np.random.uniform(-1,1,input_n)
		print(self.w)

	def calculate(self, inputVals):
		vals = np.dot(inputVals, self.w)
		vals /= np.sum(self.w)
		#print(vals)
		return vals

def main():
	net = MultiplicativeNetwork([3,3,3])
	print(net.calculate([1,2,3]))

if __name__ == '__main__':
	main()