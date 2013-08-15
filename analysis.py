import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_max_min(data):
	localMaxes = {}
	localMins = {}

	range = len(data)

	grad_prev = 0
	for i in xrange(range):
		neighbours_left = []
		neighbours_right = []
		for j in xrange(-10,0):
			if i+j > 0 and i+j < range:
				neighbours_left.append(data[i+j])
		for j in xrange(0,10):
			if i+j > 0 and i+j < range:
				neighbours_right.append(data[i+j])
		grad = 0
		for neighbour in neighbours_left:
			grad += data[i] - neighbour
		for neighbour in neighbours_right:
			grad += neighbour - data[i]
		if grad_prev > 0 and grad < 0:
			localMaxes[i] = data[i]
			#print 'Max: ' + str(i) + ' ' + str(grad_prev) + ' ' + str(grad)
		if grad_prev < 0 and grad > 0:
			localMins[i] = data[i]
			print 'Min: ' + str(i) + ' ' + str(grad_prev) + ' ' + str(grad)
		grad_prev = grad

	print 'key = location (index); value = height'
	print 'Max:'
	print localMaxes
	print 'Min:'
	print localMins

def func(x, *p):
	a,b,c,A,B,C = p
	return a*np.exp(-1 * ((x-b)**2)/(2*(c**2))) + A*np.exp(-1 * ((x-B)**2)/(2*(C**2)))

data = []
with open('hist.dat') as file:
	for line in file:
		data.append(int((line.split(','))[1]))

x = np.array( data[329:892] )

p0 = [1.,0.,1.,1.,0.,1.]

print curve_fit(func, x, y, p0=p0)

#plt.plot(x)
plt.plot(func(x, curve_fit(func, x, y, p0=p0)))
plt.show()

