import pygame
import numpy as np
from line import Line

"""
Track Segment Class
	From two simple positions generates a path connecting the two
"""
class TrackSegment:

	def __init__(self, p1, p2, width=100, detail=10, curveMagnitude=0):
		self.p1 = np.array(p1)
		self.p2 = np.array(p2)
		self.width = width
		self.detail = detail * 2
		self.curveMag = curveMagnitude * detail

	def calculateTrackPath(self):
		"""
		Calculate Track Path Function
			Interpolates between the two positions and generates a path
		"""
		v = (self.p2 - self.p1) / self.detail
		nv = np.array((-v[1], v[0]))
		nv /= np.sqrt(np.sum(nv**2))
		points = []
		for i in range(self.detail + 1):
			newp = self.p1 + v * i
			if self.curveMag != 0:
				newp += nv * (1-(abs(int(self.detail/2)-i)/(self.detail/2))**2) * self.curveMag
			points.append(newp)
		self.points = points

	def calculateTrackLines(self):
		"""
		Calculate Track Lines Function
			From the path generates an inner and outter track
		"""
		self.calculateTrackPath()
		innerPoints = []
		outerPoints = []
		for i in range(len(self.points)-1):
			nv = self.points[i+1] - self.points[i]
			nv = np.array((-nv[1], nv[0]))
			nv /= np.sqrt(np.sum(nv**2))
			innerPoints.append((self.points[i]+self.points[i+1])/2 + nv*self.width/2)
			outerPoints.append((self.points[i]+self.points[i+1])/2 - nv*self.width/2)
		self.innerPoints = innerPoints
		self.outerPoints = outerPoints
		return innerPoints, outerPoints


	def draw(self, window, debug=0):
		"""
		Draw Function
			Draws track segment
		"""
		if debug >= 1:
			pygame.draw.line(window, (120,120,120), self.p1, self.p2)
			for i in range(len(self.points)-1):
				pygame.draw.line(window, (100,255,100), self.points[i], self.points[i+1])
		for i in range(len(self.innerPoints)-1):
			pygame.draw.line(window, (100,100,255), self.innerPoints[i], self.innerPoints[i+1])
			pygame.draw.line(window, (100,100,255), self.outerPoints[i], self.outerPoints[i+1])


"""
Track Manager Class
	Handles track calculations
"""
class TrackManager:

	def __init__(self):
		self.segments = []
		self.iPoints = []
		self.oPoints = []

	def addSegment(self, segment):
		"""
		Add Segment Function
			Adds a track segment to the track manager
		"""
		self.segments.append(segment)

	def addSegments(self, segments):
		"""
		Add Segments Function
			Adds multiple segments to the track manager
		"""
		for s in segments:
			self.addSegment(s)

	def gatherTrackPoints(self):
		"""
		Gather Track Points Function
			Gathers inner and outer track points
		"""
		self.iPoints = []
		self.oPoints = []
		for seg in self.segments:
			iPoints, oPoints = seg.calculateTrackLines()
			if self.iPoints == []:
				self.iPoints = iPoints
				self.oPoints = oPoints
			else:
				self.iPoints = np.concatenate((self.iPoints, iPoints))
				self.oPoints = np.concatenate((self.oPoints, oPoints))

	def close(self):
		"""
		Close Function
			Connects first track segment to last track segment
		"""
		self.iPoints = np.concatenate((self.iPoints, [self.iPoints[0]]))
		self.oPoints = np.concatenate((self.oPoints, [self.oPoints[0]]))

	def generateLines(self):
		"""
		Generate Lines Function
			Generates the lines of the track for rendering purposes
		"""
		self.lines = []

		for i in range(len(self.iPoints)-1):
			p1 = self.iPoints[i]
			p2 = self.iPoints[i+1]
			l = Line(p1[0], p1[1], p2[0], p2[1])
			self.lines.append(l)
			p1 = self.oPoints[i]
			p2 = self.oPoints[i+1]
			l = Line(p1[0], p1[1], p2[0], p2[1])
			self.lines.append(l)

	def boundaryOcclusion(self, line):
		"""
		Boundary Occlusion
			Checks if a line passes the boundary test with each line in the track for optimization purposes
		"""
		potentialLines = []
		for l in self.lines:
			if line.boundaryTest(l):
				potentialLines.append(l)
		return potentialLines

	def checkLineIntersection(self, line):
		"""
		Check Line Intersection Function
			Checks if a line intersects the track at any point
		"""
		lines = self.boundaryOcclusion(line)
		for l in lines:
			pos = line.solveIntersection(l)
			if pos != None and line.pointCollision(l, pos):
				return True
		return False

	def lineIntersectionPosition(self, line):
		"""
		Line Intersection Position Function
			Checks if a line intersects the track at any point and provides that point
		"""
		lines = self.boundaryOcclusion(line)
		positions = []
		for l in lines:
			pos = line.solveIntersection(l)
			if pos != None and line.pointCollision(l, pos):
				positions.append(pos)
		return positions

	def draw(self, window, debug=0):
		"""
		Draw Function
			Draws the entire track
			Use debug >= 1 to show track segment connections
		"""
		if debug >= 1:
			for s in self.segments:
				pygame.draw.line(window, (100,255,100), s.p1, s.p2)
		for i in range(len(self.iPoints)-1):
			pygame.draw.line(window, (100,100,255), self.iPoints[i], self.iPoints[i+1])
			pygame.draw.line(window, (100,100,255), self.oPoints[i], self.oPoints[i+1])


