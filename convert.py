from struct import *
import numpy as np
import Image
import sys

#reads file header for dimensions of image
def _get_dims(file):
	header = file.read(17)
	width = int((header.split('_'))[3])
	height = int((header.split('_'))[4])
	return (width, height)
	

#returns a 2 dimensional numpy array from given .sli file
def raw2Numpy(filename):
	if not filename.endswith('.sli'):
		raise IOError('Bad filetype')
	
	with open(filename, 'rb') as file:
		array = np.empty( _get_dims(file), dtype = float )
		
		maxfp = 0							#for normalisation of image
		fp_index = 0
		fp = unpack('f', file.read(4))		#NB. unpack returns a tuple, hence fp[0]
		while fp_index < width * height - 1:
			#parsing data and setting up for normalisation
			if fp[0] <= 0:
				fp = (0,0)
			else:
				if fp[0] > maxfp:
					maxfp = fp[0]
			
			array[fp_index // width][fp_index % width] = fp[0]
		
			fp = unpack('f', file.read(4))
			fp_index += 1

		#normalising the array
		array /= maxfp
		array *= 255
		
		return array


if __name__=='__main__':
	if len(sys.argv) < 2:
		raise IOError('No filename specified')
	raw2Numpy(sys.argv[1])
			
			
