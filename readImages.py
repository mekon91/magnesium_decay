
import Image,os,sys
from numpy import *
from os import listdir
from os.path import isfile, join



file_extension = '.png'
path = "/afs/desy.de/user/d/dariep/Desktop/MagDecay/Images"

#returns all the files at path that end with the extension file_extension
def get_image_files(path,file_extension):
	return [ f for f in listdir(path) if isfile(join(path,f)) and f.endswith(file_extension)]

#returns an array with the grayscale of the image file img
def img_to_array(img):
	size = img.size
	pixels = img.load()
	
	#declare array a with same size as the image img
	a = zeros((size[0],size[1]), dtype = int) 
	
	for x in xrange(size[0]):
		for y in xrange(size[1]):
			#Note: this array contains the red values [0 255]
			#It assumes image is already grayscale!
			#TODO converter from RGBA -> grayscale
			a[y][x] = pixels[x,y][0]
	return a	

image_files = get_image_files(path,file_extension)



image_files.sort()
print "Images in folder: "
for i in image_files:
	print i,
print '\n'

try:
	i = 0
	#get dimension of the image. It assumes all images are the same resolution
	first_img = Image.open
	
	
	m = zeros((len(image_files),10,10), dtype = int)
	#m = m.astype(int)
	for file in image_files:
		print "Opening image " + file
		print path + "/" + file
		img = Image.open(path + "/" + file)
		a = img_to_array(img)
		m[i] = a
		i += 1
	print m	
except IOError:
	print "Problems opening files!"
	

		
