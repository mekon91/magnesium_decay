#NB. This script for finding the threshold values is designed to be edited before running


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import convert, helpFunctions

#definition of function to fit
def gauss(i, x, *p):
	a = p[3*i - 3]
	b = p[3*i - 2]
	c = p[3*i - 1]
	return a*np.exp(-1 * ((x-b)**2)/(2*(c**2)))
def func(x, *p):
	ret = 0
	for i in xrange(len(p) // 3):
		ret += gauss(i, x, *p)
	return ret
	
#scales the gaussian. i is index of gaussian - e.g. 2 for 'gauss2'
def get_n(i, *p):
	return 1 / (p[3*i - 3] * math.sqrt(2 * math.pi * p[3*i - 1]**2))

#returns intersections of gaussians with supplied indices
def get_intersection(i,j,*p):
	#declaring variables 
	a1 = abs(p[3*i - 3])
	b1 = abs(p[3*i - 2])
	c1 = abs(p[3*i - 1]) ** 2
	a2 = abs(p[3*j - 3])
	b2 = abs(p[3*j - 2])
	c2 = abs(p[3*j - 1]) ** 2
	#positive solution 
	return [ (2*c2*b1 - 2*c1*b2 - math.sqrt((2*c1*b2 - 2*c2*b1)**2 - 4 * (c2 - c1)*(b1*b1*c2 - b2*b2*c1 - 2*c1*c2*math.log(math.sqrt(c2/c1)))))/(2 * (c2 - c1)), (2*c2*b1 - 2*c1*b2 + math.sqrt((2*c1*b2 - 2*c2*b1)**2 - 4 * (c2 - c1)*(b1*b1*c2 - b2*b2*c1 - 2*c1*c2*math.log(math.sqrt(c2/c1)))))/(2 * (c2 - c1)) ]

#functions used for parse_data method
def find_background(data):
	#return np.mean(data[len(data)//2:])
	return 1.25*data[3 * np.argmax(data)]
def find_min(data):
	ret = []
	for i in xrange(len(data) - 1):
		if data[i] < 0 and data[i+1] > 0:
			ret.append(i)
	return ret
def find_max(data):
	ret = []
	for i in xrange(len(data) - 1):
		if data[i] > 0 and data[i+1] < 0:
			ret.append(i)
	return ret
	
def parse_data(y):
	#convert to numpy array
	y = np.array(y)
	#background is found by taking value at index 3 * index of air peak (assumed to be highest peak in data).
	#This is then multiplied by 1.25 to ensure the background is completely removed 
	#and the gaussians we are trying to fit cross zero.
	y -= find_background(y)
	#left hand edge of region of interest is assumed to be the index where the modified data goes from 
	#negative to positive.
	print find_min(y)
	y = y[find_min(y)[-1]:]
	#right hand edge of region of interest is assumed to be the index where the modified data goes from 
	#positive to negative.
	y = y[:find_max(y)[-1]]
	return y

#paths defined for our local work environment
path1 = '/space/Data/Data1/reco/'
path0 = '/space/Data/Data0/reco/'

#Console zero: set data source and initial parameters for fitting
#p0 is the initial guess for gaussian parameters. See 'get_threshold' method comments for description.
"""mode: tiff edges

print 'mode: tiff edges'
y = np.array( [float(line.strip()) for line in open('hist.txt')] )
p0 = [15000.,24.,1., 12000.,158.,1., 16000.,191.,1.]
x = np.array( range( len(y) ) )
"""

"""mode: tiff trimmed

print 'tiff trimmed'
y = np.array( [float(line.strip()) for line in open('hist2.txt')][100:] )
p0 = [28000.,70.,1., 95000.,60.,1.]
x = np.array( range(len(y)) )
"""

"""mode: tiff trimmed and chopped

print 'tiff'
y = np.array( [float(line.strip()) for line in open('hist2.txt')] )
y = parse_data(y)

p0 = [float(np.amax(y)),0.,1., float(np.amax(y)),float(len(y)),1.]
x = np.array( range(len(y)) )

#p0 = [28000.,30.,1., 95000.,20.,1.]
"""

"""mode: float 256

print 'float 256'
numBins = 256
y = convert.raw_to_histogram(path1, numBins)
p0 = [320000.,15.,1., 1.5e6, 18.,1., 1.4e7,19.,1., 12000.,67.,1., 37000.,78.,1.]
x = np.array( range( len(y) ) )
"""

"""mode: float 2048
"""
print 'float 2048'
numBins = 2048
y = convert.raw_to_histogram(path1, numBins)
y = parse_data(y)
#p0 = [float(np.amax(y)),0.,1., float(np.amax(y)),float(len(y)),1.]
p0 = [7700.,58.,1., 36000.,150.,1.]

x = np.array( range( len(y) ) )


"""mode: float 2048

print 'float 2048 chopped'
numBins = 2048
y = np.array( convert.raw_to_histogram(path1, numBins)[450:697] ) - 2900
x = np.array( range( len(y) ) )
p0 = [float(np.amax(y)),0.,1., float(np.amax(y)),float(len(y)),1.]
"""

#Console one: region of interest
#----------------------------------------------------------------------------------------------------
#region =  [370,800]
region = [0, len(y)]
gausses = []			#NB gausses is a list of gauss indices to plot. If it is empty, all gausses will be plotted
#----------------------------------------------------------------------------------------------------


#NB, i think these functions have to be here because python only supports passing 1 unknown-length array at once.
def plot_gausses():
	if len(gausses) == 0:
		for i in xrange(len(p) // 3):
			plt.plot( gauss(i, x, *p) )
	else:
		for i in xrange(len(gausses)):
			plt.plot( gauss(gausses[i], x, *p) )
			print gausses[i]
		
def plot_normalised_gausses():
	if len(gausses) == 0:
		for i in xrange(len(p) // 3):
			plt.plot( get_n(i, *p) * gauss(i, x, *p) )
	else:
		for i in xrange(len(gausses)):
			plt.plot( get_n(gausses[i], *p) * gauss(gausses[i], x, *p) )



#Console two: style of output
#----------------------------------------------------------------------------------------------------
p = curve_fit(func, x, y, p0=p0) [0]
print p
#plt.plot( y )
#plt.plot( func(x, *p) )
#plot_gausses()
plot_normalised_gausses()
print "Intersection of normalised gausses: " + str(get_intersection(1,2,*p))
#print "Midpoint of gausses: " + str( (p[1] + p[4]) / 2 )
#print "Intersection of normalised gausses: " + str(get_intersection(2,3,*p))
print "Threshold 0: " + str( p[1] - 6 * p[2] )
#----------------------------------------------------------------------------------------------------

#plt.xticks([i for i in x if i%10 == 0])
#plt.xlim(region)
plt.show()


#automated method with nicer output
def get_threshold(data):
	#convert to numpy array
	data = np.array(data)
	#background is found by taking value at index 3 * index of air peak (assumed to be highest peak in data).
	#This is then multiplied by 1.25 to ensure the background is completely removed 
	#and the gaussians we are trying to fit cross zero.
	data -= find_background(data)
	#left hand edge of region of interest is assumed to be the index where the modified data goes from 
	#negative to positive.
	left = find_min(data)[-1]
	data = data[left:]
	#right hand edge of region of interest is assumed to be the index where the modified data goes from 
	#positive to negative.
	right = find_max(data)[-1]
	data = data[:right]
	
	#set p0 to a good guess for gaussians. p0[0] is height, p0[1] is position, p0[2] is width. And etc for 2nd gaussian
	p0 = [7700.,58.,1., 36000.,150.,1.]
	#sometimes the below will suffice as a good guess
	#p0 = [float(np.amax(data)),0.,1., float(np.amax(data)),float(len(data)),1.]
	x = np.array( range(len(data)) )
	p = curve_fit(func, x, data, p0=p0) [0]
	plt.plot( data )
	plt.plot( func(data, *p) )
	
	#this prints [corrosive medium-magnesium threshold bin, corroded magnesium-uncorroded magnesium threshold guess 1, corroded magnesium-uncorroded magnesium threshold guess 2]
	#there are 2 guesses for 2nd threshold because the gaussians intersect at 2 points
	print [left + (p[4] - 6 * p[5]), left + get_intersection(1,2,*p)[0], left + get_intersection(1,2,*p)[1]]
	plt.show()


#get_threshold(y)
