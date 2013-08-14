from os import listdir
from os.path import isfile, join
from struct import *

path = '/space/hzg09c_rawbin2/reco/'
#path = './'

def raw_to_frequencies(path):
	files = [ f for f in listdir(path) if isfile(join(path,f)) and f.endswith('.sli')]
	
	numBins = 2048
	
	fpmax = 0.0153947146609
	fpmin = -0.00610217032954
	
	freq_list = [0]*numBins
	binSize = (fpmax-fpmin) / (numBins-1)
	n = 0
	for f in files:
		if files.index(f) > 159 and files.index(f) < 260:
			print str(n) + '/' + str(260)
			with open(path + f, 'rb') as file:
				file.read(17)
				for i in xrange(775*775):
					fp = unpack('f', file.read(4))
					k = int((fp[0]-fpmin) / binSize)
					
					freq_list[k] += 1
		n += 1
		
	print freq_list
	f = open('freq.dat', 'w')
	i = 0
	for v in freq_list:
		f.write(str(i) + ',' + str(v) + '\n')
		i += 1
	f.close()
	
raw_to_frequencies(path)
	
	
