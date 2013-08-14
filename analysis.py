import numpy as np
import matplotlib.pyplot as plt

data = []
with open('freq.dat') as file:
	for line in file:
		data.append(int((line.split(','))[1]))
	
localMaxes = {}
localMins = {}

data = data[765:1042]

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
		print 'Max: ' + str(i) + ' ' + str(grad_prev) + ' ' + str(grad)
	if grad_prev < 0 and grad > 0:
		localMins[i] = data[i]
		print 'Min: ' + str(i) + ' ' + str(grad_prev) + ' ' + str(grad)
	grad_prev = grad

print 'key = location (index); value = height'
print 'Max:'
print localMaxes
print 'Min:'
print localMins

plt.plot(data)
plt.show()
