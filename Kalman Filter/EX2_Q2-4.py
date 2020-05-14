##
# Main function of the Python program.
#
##

import pandas as pd
import numpy as np

#Fill in Those functions:
def dot3(a,b,c):
    return (a.dot(b)).dot(c)

def rmse(predictions, targets):
	return np.sqrt(((np.array(predictions) - np.array(targets)) ** 2).mean())
	
def computeRMSE(trueVector, EstimateVector):
	return list(map(rmse,trueVector.transpose(),EstimateVector.transpose()))
	

def computeRadarJacobian(Xvector, THRESH = 0.0001, ZERO_REPLACEMENT = 0.0001):
	
	px=Xvector[0]
	py=Xvector[1]
	vx=Xvector[2]
	vy=Xvector[3]
	
	d_squared = px * px + py * py 
	d = np.sqrt(d_squared)
	d_cubed = d_squared * d
	
	if d_squared < THRESH:
 
		print("WARNING: in calculate_jacobian(): d_squared < THRESH")
		H = np.matrix(np.zeros([3, 4]))
 
	else:

		r11 = px / d
		r12 = py / d
		r21 = -py / d_squared
		r22 = px / d_squared
		r31 = py * (vx * py - vy * px) / d_cubed
		r32 = px * (vy * px - vx * py) / d_cubed
	
		H = np.array([[r11, r12, 0, 0], 
									[r21, r22, 0, 0], 
									[r31, r32, r11, r12]])

	return H
	
def computeCovMatrix(dt, x, y):
	
	dt2 = dt * dt
	dt3 = dt * dt2
	dt4 = dt * dt3
	
	
	
	r11 = dt4 * x / 4
	r13 = dt3 * x / 2
	r22 = dt4 * y / 4
	r24 = dt3 * y /	2
	r31 = dt3 * x / 2 
	r33 = dt2 * x
	r42 = dt3 * y / 2
	r44 = dt2 * y
	
	Q = np.array([[r11, 0, r13, 0],
								[0, r22, 0, r24],
								[r31, 0, r33, 0], 
								[0, r42, 0, r44]])
	
	return Q
	
    
def computeFmatrix(deltaT):
    return np.array([[1,0,deltaT,0],[0,1,0,deltaT],[0,0,1,0],[0,0,0,1]])

def stateSpace2sensorSpace(Xvector,THRESH = 0.0001):
	px=Xvector[0]
	py=Xvector[1]
	vx=Xvector[2]
	vy=Xvector[3]
	
	
	
	rho = (px**2+py**2)**0.5
	phi = np.arctan2(py, px)
	
	
	if rho < THRESH :
		rho, phi, rhodot = 0, 0, 0
	else :
		rhodot = (px*vx+py*vy)/rho
		
	
	return np.array([rho, phi, rhodot])
  

  
def main(useRadar, usenoise):
	my_cols = ["A", "B", "C", "D", "E","f","g","h","i","j","k"]
	data = pd.read_csv("obj_pose-laser-radar-synthetic-input.txt", names=my_cols, delim_whitespace = True, header=None)
	P = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1000.,0.],[0.,0.,0.,1000.]])
	sigma_aX = 9. 
	sigma_aY =9.
	xEstimate = []
	xTrue = []
	R_lidar = np.array([[0.0225, 0.0],[0.0, 0.0225]])
    
	R_radar = np.array([[0.09, 0, 0],[0.0, 0.0009, 0],[0, 0, 0.09]]) 
	H = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])
	I = np.eye(4)
	
	
	firstMeasurment = data.iloc[0,:].values
	X_state_current = np.array([float(firstMeasurment[1]),float(firstMeasurment[2]),0,0])
	X_true_current = np.array([float(firstMeasurment[4]),float(firstMeasurment[5]),0,0])
	xEstimate.append(X_state_current)
	xTrue.append(X_true_current)
	timeStamp = firstMeasurment[3]
    #fill in X_true and X_state. Put 0 for the velocities
	
	for i in range(1,len(data)):
		currentMeas = data.iloc[i,:].values
		if(currentMeas[0]=='L'):
			
			#update 
			deltaT = (currentMeas[3]- timeStamp)/1000000
			F_matrix = computeFmatrix(deltaT)
			timeStamp = currentMeas[3]
			Q = computeCovMatrix(deltaT, sigma_aX, sigma_aY)
			
            #perfrom predict

			X_state_current = (F_matrix.dot(X_state_current)) 
			X_true_current = np.array([float(currentMeas[4]),float(currentMeas[5]),float(currentMeas[6]),float(currentMeas[7])])
			
			#print(P.shape)
			if usenoise == True:
				P  = dot3(F_matrix , P , np.transpose(F_matrix)) + Q   
			else :
				P  = dot3(F_matrix , P , np.transpose(F_matrix)) 

            #pefrom measurment update
			z = np.array([float(currentMeas[1]),float(currentMeas[2])]) 
			y = np.transpose(z) - (H.dot(X_state_current))
			S = dot3(H , P , np.transpose(H)) + R_lidar
			K = dot3(P , np.transpose(H) , np.linalg.inv(S))	
			
			
			X_state_current = X_state_current + (K.dot(y))
			P  = (I - K.dot(H)).dot(P)

			
		if(currentMeas[0]=='R' and useRadar):
			
            #update 
			deltaT = (currentMeas[4]- timeStamp)/1000000
			F_matrix = computeFmatrix(deltaT)
			Q = computeCovMatrix(deltaT, sigma_aX, sigma_aY)
			timeStamp = currentMeas[4]
			X_true_current = np.array([float(currentMeas[5]),float(currentMeas[6]),float(currentMeas[7]),float(currentMeas[8])]) 
			
			#perfrom predict
			X_state_current = (F_matrix.dot(X_state_current)) 	
			
			if usenoise == True:
				P  = dot3(F_matrix , P , np.transpose(F_matrix)) + Q   
			else :
				P  = dot3(F_matrix , P , np.transpose(F_matrix))
				

            #pefrom measurment update
			jacobian = computeRadarJacobian(X_state_current)
			z = np.array([float(currentMeas[1]),float(currentMeas[2]),float(currentMeas[3])]) 
			Xconvert = stateSpace2sensorSpace(X_state_current)			
			y = np.transpose(z) - Xconvert

			S = dot3(jacobian , P , np.transpose(jacobian)) + R_radar
			K = dot3(P , np.transpose(jacobian) , np.linalg.inv(S))
			
			
			X_state_current = X_state_current + (K.dot(y))
			P  = (I - K.dot(jacobian)).dot(P)
		   
		xEstimate.append(X_state_current)
		xTrue.append(X_true_current)
	
	rmse = computeRMSE(np.array(xEstimate), np.array(xTrue) )
	print(rmse)

        
    

if __name__ == '__main__':
	print('Q2:')
	main(False,False)
	print('Q3:')
	main(False,True)
	print('Q4:')
	main(True,True)
