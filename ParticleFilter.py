import pandas as pd
import os
import numpy as np
from dataclasses import dataclass
import random
import time




np.random.seed(14)

@dataclass
class Particle:
	x: float
	y: float
	theta: float
	weight: float 
		


@dataclass
class LandMark:
	x: float
	y: float
	index: int
	
gt_data = pd.read_csv('data/gt_data.txt', names=['X','Y','Orientation'], sep=' ')
map_data = pd.read_csv('data/map_data.txt', names=['X','Y','# landmark'])
control_data = pd.read_csv('data/control_data.txt', names=['velocity','Yaw rate'], sep=' ')

#observation = pd.read_csv('data/observation/observations_000001.txt', names = ['X cord','Y cord'], sep=' ')


result = [(x,y, landmark) for x,y,landmark in zip(map_data['X'],map_data['Y'], map_data['# landmark'])]
landarkList=[]
for res in result:
	l = LandMark(res[0],res[1],res[2])
	landarkList.append(l)

	

a = os.listdir("data/observation")
a.sort()
observation=[]
for i in range(len(a)):
	fileName = 'data/observation/'+a[i]
	observationTmp = pd.read_csv(fileName, names = ['X cord','Y cord'], sep=' ')
	observation.append(observationTmp)	




	
def calculateDistance(landmark1, landmark2):
	a =  np.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)
	return a
	
 #____________Q1______________________	
	
def findClosestLandmark(map_landmarks, singleObs):
	for i in singleObs:
		min = np.inf
		for j in map_landmarks:
			d = calculateDistance(i,j)
			if d < min :
				i.index = j.index
				min = d		
	return 	singleObs	

	
#__________________________________		
	
		
def getError(gt_data, bestParticle):
	error1 = np.abs(gt_data[0] - bestParticle.x)
	error2 = np.abs(gt_data[1] - bestParticle.y)
	error3 = np.abs(gt_data[2] - bestParticle.theta)
	if(error3>2*np.pi):
		error3 = 2*np.pi - error3
	return (error1, error2, error3)

def findObservationProbability(closest_landmark,map_coordinates, sigmaX, sigmaY):
	
	mew_x = closest_landmark.x
	mew_y = closest_landmark.y
	
	x = map_coordinates.x;
	y = map_coordinates.y;
	

	weight1 = (x-mew_x)**2/(2*(sigmaX)**2) + (y-mew_y)**2/(2*sigmaY**2)	
	ans = np.exp(-weight1)
	return ans



def mapObservationToMapCoordinates(observation, particle):
	x = observation.x
	y = observation.y

	xt = particle.x
	yt = particle.y
	theta = particle.theta

	MapX = x * np.cos(theta) - y * np.sin(theta) + xt
	MapY = x * np.sin(theta) + y * np.cos(theta) + yt

	return MapX, MapY

def mapObservationsToMapCordinatesList(observations, particle):
	
	convertedObservations=[]
	i=0
	for obs in observations.iterrows():
		singleObs = LandMark(obs[1][0],obs[1][1],1)
		mapX, mapY = mapObservationToMapCoordinates(singleObs, particle)
		tmpLandmark = LandMark(x=mapX, y=mapY, index=i)
		i+=1
		convertedObservations.append(tmpLandmark)
	return convertedObservations	


class ParticleFilter:
	particles = []
	weights = np.array([])
	def __init__(self, intialX, initialY, std, numOfParticles):
		print("init")
		self.number_of_particles = numOfParticles
		self.weights = np.array([1.]*numOfParticles)
		
		for i in range(self.number_of_particles):
			x = random.gauss(intialX, std)
			y = random.gauss(initialY, std)
			theta = random.uniform(0, 2*np.pi)
			tmpParticle = Particle(x,y , theta,1.)
			self.particles.append(tmpParticle)
			
			
			
	def moveParticles(self, velocity, yaw_rate, delta_t=0.1):
		for i in range(self.number_of_particles):
			if(yaw_rate!=0):
				theta = self.particles[i].theta
				newTheta = (theta + delta_t * yaw_rate)%(2 * np.pi);
				newX =  self.particles[i].x + (velocity/yaw_rate)*(np.sin(newTheta)-np.sin(theta));
				newY =  self.particles[i].y + (velocity/yaw_rate)*(np.cos(theta)-np.cos(newTheta));
				
				#todo Add noise!!
				self.particles[i].x = newX + random.gauss(0, 0.03)
				self.particles[i].y = newY + random.gauss(0, 0.03)
				self.particles[i].theta = newTheta +random.gauss(0, 0.01)
			else:
				print("ZERO!!!")
			

#____________Q2______________________	
		
	def UpdateWeight(self, observations):

		for i,p in enumerate(self.particles):

			#1. transform observations from vehicle to map coordinates assuming it's the particle observing
			ConvObservations = mapObservationsToMapCordinatesList(observations, p)

			landmarks_in_range = []
			#2.find landmark in the sensor's range
			for l in landarkList:
				d = calculateDistance(p,l)
				if d < sensor_range :
					landmarks_in_range.append(l)
		
			#3. find which landmark is likely being observed based on nearest neighbor method
			ob_landmarks = findClosestLandmark(landmarks_in_range,ConvObservations)

			#4. determine the weights based difference particle's observation and actual observation
			particle_likelihood = 1.
			norm_factor =  (2 * np.pi * (sigmaX * sigmaY) )
			
			for ob in ob_landmarks:
				ido =  ob.index
				prob = findObservationProbability(ob , landarkList[ido - 1],sigmaX, sigmaY)
				)
				particle_likelihood = particle_likelihood * prob / norm_factor	
			self.weights[i] = particle_likelihood
		#normalize weights
		norm_factor = sum(self.weights)
		for w in range(self.number_of_particles):
			self.weights[w] = self.weights[w] / (norm_factor)
			self.particles[w].weight = self.weights[w]
		print("weight sum",sum(self.weights))	
	
#__________________________________	
	
	def getBestParticle(self):
		best_particle =  max(self.particles, key= lambda particle: particle.weight)
		return best_particle	
	
	def getBestParticleOut(self):
		x=0
		y=0
		theta=0
		for i in range(self.number_of_particles):
			x+= self.particles[i].x
			y+= self.particles[i].y
			theta+= self.particles[i].theta
		x=x/self.number_of_particles
		y=y/self.number_of_particles
		theta=theta/self.number_of_particles
		best_particle =  Particle(x,y,theta, weight=1)
		return best_particle		
	
	def PrintWeights(self):
		for i in range(self.number_of_particles):
			print("Weight:",self.particles[i].weight, self.particles[i].x,self.particles[i].y)

#____________Q3______________________	
	def Resample(self):
		newParticles = []
		maxweight = max(self.weights)
		index = int(random.uniform(0, self.number_of_particles))
		b=0
		for i in range(self.number_of_particles):
			b = b + random.uniform(0, 2*maxweight)
			windex = self.weights[index]
			if windex < b :
				b = b - windex
				index = (index + 1) % self.number_of_particles
			newParticles.append(self.particles[index])
			self.weights[index] = self.particles[index].weight	
		self.particles = newParticles

#____________________________________

		
sigmaY = 3
sigmaX = 3
sensor_range = 50
magicNumberOfParticles = 200


start = time.time()

def main():
	#particleFilter = ParticleFilter(0 ,0 ,100, numOfParticles=magicNumberOfParticles)
	particleFilter = ParticleFilter(6.2785 ,1.9598 ,0.3, numOfParticles=magicNumberOfParticles)
	for i in range(len(observation)):

		if(i!=0):
			velocity = control_data.iloc[i-1][0]
			yaw_rate = control_data.iloc[i-1][1]
			particleFilter.moveParticles(velocity, yaw_rate)
		a = observation[i].copy()
		particleFilter.UpdateWeight(a)
		particleFilter.Resample()
		bestP = particleFilter.getBestParticle()
		error = getError(gt_data.iloc[i], bestP)
		print(i,error)
	end = time.time()
	print(end - start)	

main()	