from struct import *
import numpy as np
import sys, Image, helpFcts
from os import listdir
from os.path import isfile, join

 
		 
#reads file header for dimensions of image
def _get_dims(file):
	header = file.read(17)
	width = int((header.split('_'))[3])
	height = int((header.split('_'))[4])
	return (width, height)
	

#returns a 2 dimensional numpy array from given .sli file
def read_sli(filename):
	
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



#returns all the files at path that end with the extension file_extension
def _get_files(path,file_extension):
	try:
		files = [ f for f in listdir(path) if isfile(join(path,f)) and f.endswith(file_extension)]
	except IOError:
		"No such file or directory: "  + path 	
	#check if there are files in folder and print their names out
	if (len(files) >= 1):
		
		print "Files at " + path + " with extension " + file_extension + " :" 
		for i in files:
			print i,
		print '\n'
		helpFcts.sort_nicely(files)	
		return files
	else:
		print "No files in folder " + path
		return None

#using the _get_files, it then reads all images at path with the extension 'file_extension', and stacks them on a 3D array m. m[i] is the ith image
#The array m entries are [0 255] 
def image_from_folder_to_array(path, file_extension):
	image_files = _get_files(path,file_extension)
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

#similarly to image_from_folder_to_array, this method reads the .sli files into an array. The array entries are float numbers
def sli_from_folder_to_array(path):
	files = _get_files(path,".sli")
	try:
		i = 0
		#get dimension of the first file. Assumes all files have same header (thus same width,height)
		
		first_file = open(path + "/" +files[0], 'rb')
		size = _get_dims(first_file)	
		m = np.zeros((len(files),size[0],size[1]), dtype = float)
			
		for file in files:
			print "Opening file at " + path + "/" + file	
			m[i] = read_sli(path + "/" + file)							
			i += 1
		print 
		return m		
	except IOError:
		print "Problems opening files from" + path
		return None	



if __name__=='__main__':
	
	if len(sys.argv) < 2:
		raise IOError('No filename specified')
		
	raw2Numpy(sys.argv[1])
