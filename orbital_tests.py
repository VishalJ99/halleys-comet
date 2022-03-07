import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

'''
todo: Verify why total energy is not constant
'''

def plot_total_energy(results,mass_array): 
	''''
	Takes in a array of results and plots total energy of system against time

	Parameters
	----------
	results : np.ndarray
		array of results, each row has entries [t,x0,y0,u0,v0,...,xN,yN,uN,vN], where t is time, x,y are positions and u,v are x y velocities. Assumes times, positions and velocities are in units of meters and seconds
	
	mass_array: np.ndarray
		array of masses for each body. Assumes mass is in kilograms.
	
	Returns:
	--------
	KE : np.ndarray
		array of kinetic energies in joules of each body at every time, eg row [t, T0, T1, ..., TN] of array

	GPE : np.ndarray
		array of potential energies in joules of each body at every time, eg row [t, U0, U1, ..., UN] of array 
	'''
	# define G in units of meters, kilograms and seconds
	G = 6.674e-11
	
	# find number of bodies
	N = len(mass_array)
	assert (results.shape[1]-1) / 4 == N, '[ERROR] mismatch between length of mass array and number of cols in results array'

	# arrays to store energy of system at each point in time
	KE = np.zeros((len(results),N))
	GPE = np.zeros((len(results),N))

	# calculate energies for each body
	for i in range(N):
		# calculate kinetic energy, T = 0.5 * m * (vx **2 + vy **2)
		m_i = mass_array[i]
		T = 0.5 * m_i * (results[:,4*i+3]**2 + results[:,4*i+4]**2)
		ic(T.shape)
		# calculate potential energy, U_i = - sum j (j =/= i) G mi * mj / |rij|
		U = np.zeros(len(results))
		xi, yi = results[:,4*i+1], results[:,4*i+2]
		for j in range(N):
			if i!=j:
				xj, yj = results[:,4*j+1], results[:,4*j+2]
				xij, yij = xi - xj, yi - yj
				m_j = mass_array[j]
				rij = (xij**2 + yij**2)**0.5
				U += -G*m_j*m_i / rij
		
		# store results for mass i
		KE[:,i] = T
		GPE[:,i] = U

	# plot results
	system_KE = np.sum(KE,axis=1)
	system_GPE = np.sum(GPE,axis=1)
	time = results[:,0]
	plt.title('Energy of system')
	plt.xlabel('time (s)')
	plt.ylabel('Energy (J)')
	plt.plot(time, system_KE , label = 'KE of system')
	plt.plot(time, system_GPE, label = 'GPE of system')
	plt.plot(time, system_KE + system_GPE , label = 'total energy of system')
	plt.legend(loc="upper left")
	plt.show()


