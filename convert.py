from struct import *
import numpy as np
import sys, Image, helpFunctions
		 
#reads file header for dimensions of image
def _get_dims(file):
	header = file.read(17)
	width = int((header.split('_'))[3])
	height = int((header.split('_'))[4])
	return (width, height)
	

#returns a 2 dimensional numpy array from given .sli file
def raw_to_numpy2D(filename):
	
	if not filename.endswith('.sli'):
		raise IOError('Bad filetype: .sli file required')
	
	with open(filename, 'rb') as file:
		array = np.empty( _get_dims(file), dtype = float )
		fp_index = 0
		fp = unpack('f', file.read(4))		#NB. unpack returns a tuple, hence fp[0]
		
		while fp_index < array.shape[0] * array.shape[1] - 1:			
			array[fp_index // array.shape[0]][fp_index % array.shape[0]] = fp[0]
			fp = unpack('f', file.read(4))
			fp_index += 1
		
		return array

#using the helpFunctions.get_files, it then reads all images at path with the extension 'file_extension', and stacks them on a 3D array m. m[i] is the ith image
#The array m entries are [0 255] 
def images_to_numpy3D(path, file_extension):
	image_files = helpFunctions.get_files(path,file_extension)
	try:
		i = 0
		#get dimension of the image. It assumes all images are the same resolution
		first_img = Image.open(path + "/" + image_files[0])
			
		m = np.zeros((len(image_files),first_img.size[0],first_img.size[1]), dtype = int)
			
		for file in image_files:
			print "Opening image at " + path + "/" + file
			
			img = Image.open(path + "/" + file)						
			#convert the image to a 2D array. flip the x axis, because for some reason the image img is mirrored. 
			#The flipud ( mirror x ) should generaly not be required!
			a = np.flipud(np.asarray(img))		
			m[i] = a
			i += 1
		print
		return m		
	except IOError:
		print "Problems opening files from" + path
		return None	

#similarly to images_to_3Dnumpy, this method reads the .sli files into an array. The array entries are float numbers
def raw_to_numpy3D(path):
	files = helpFunctions.get_files(path,".sli")
	try:
		i = 0
		#get dimension of the first file. Assumes all files have same header (thus same width,height)
		
		first_file = open(path + "/" +files[0], 'rb')
		size = _get_dims(first_file)	
		m = np.zeros((len(files),size[0],size[1]), dtype = float)
			
		for file in files:
			print "Opening file at " + path + "/" + file	
			m[i] = raw_to_numpy2D(path + "/" + file)							
			i += 1
		print 
		return m		
	except IOError:
		print "Problems opening files from" + path
		return None

def numpy3D_to_histogram(m):
	f = np.zeros(512,dtype = int)
	#get max and min
	minimum = m[0,0,0]
	maximum = m[0,0,0]
	print m[0,0,0]
	 
	for i in xrange(m.shape[0]):
		print "Determining max and min " + str(i) + "% "
		for j in xrange(m.shape[1]):
			for k in xrange(m.shape[2]):
				if (m[i,j,k] < minimum):
					minimum = m[i,j,k]				
				if (m[i,j,k] > maximum):
					maximum = m[i,j,k]
	print "minimum is " + str(minimum) + "\n maximum is " + str(maximum)
	
	bin_size = (maximum - minimum) / 512.
	print "Bin size is " + str(bin_size)
	for i in xrange(m.shape[0]):
		print "Float frequencies " + str(i) + "% "
		for j in xrange(m.shape[1]):
			for k in xrange(m.shape[2]):
				index = abs(int(m[i,j,k] / bin_size))
				f[index] += 1 
	return f

def _get_parameters(path, files):
	fpmin = None
	fpmax = None
	dims = None
	with open(path + files[0], 'rb') as file:
		dims = _get_dims(file)
		fp = unpack('f', file.read(4))
		fpmax = fp[0]
		fpmin = fp[0]
	n = 0
	top = str(len(files))
	for f in files:
		print str(n) + '/' + top
		with open(path + f, 'rb') as file:
			file.read(17)
			for i in xrange(dims[0]*dims[1]):
				fp = unpack('f', file.read(4))
				if fp[0] > fpmax:
					fpmax = fp[0]
				else:
					if fp[0] < fpmin:
						fpmin = fp[0]
		n += 1
	return ( dims, (fpmin, fpmax) )
		
def raw_to_histogram(path, numBins, output):
	files = helpFunctions.get_files(path,'.sli')
	
	print 'loading params'
	params = _get_parameters(path, files)
	
	width = (params[0])[0]
	height = (params[0])[1]
	fpmin = (params[1])[0]
	fpmax = (params[1])[1]
	
	print 'compiling histogram'
	freq_list = [0]*numBins
	binSize = (fpmax-fpmin) / (numBins-1)
	n = 0
	top = str(len(files))
	for f in files:
		print str(n) + '/' + top
		with open(path + f, 'rb') as file:
			file.read(17)
			for i in xrange(width * height):
				fp = unpack('f', file.read(4))
				k = int((fp[0]-fpmin) / binSize)
				freq_list[k] += 1
		n += 1
		
	print freq_list
	f = open(output, 'w')
	i = 0
	for v in freq_list:
		f.write(str(i) + ',' + str(v) + '\n')
		i += 1
	f.close()

