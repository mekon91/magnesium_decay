
import Image,os,sys, scipy
import numpy as np
from os import listdir
from os.path import isfile, join
import scipy.ndimage as im
import warnings




file_extension = '.png'
path = "/afs/desy.de/user/d/dariep/Desktop/magnesium_decay/Images"

#returns all the files at path that end with the extension file_extension
def get_image_files(path,file_extension):
	try:
		image_files = [ f for f in listdir(path) if isfile(join(path,f)) and f.endswith(file_extension)]
	except IOError:
		"No such file or directory: "  + path 	
	#check if there are files in folder and print their names out
	if (len(image_files) >= 1):
		
		print "Images in folder: "
		for i in image_files:
			print i,
		print '\n'
	else:
		print "No files in folder " + path
	return image_files	
	
#returns an array with the grayscale of the image file img
def img_to_array(img):
	size = img.size
	pixels = img.load()
	
	#declare array a with same size as the image img
	a = np.zeros((size[0],size[1]), dtype = int) 
	
	for x in xrange(size[0]):
		for y in xrange(size[1]):
			#Note: this array contains the red values [0 255]
			#It assumes image is already grayscale!
			#TODO converter from RGBA -> grayscale
			a[y][x] = pixels[x,y][0]
	return a	

def read_from_file_to_array(path,file_extension):
	image_files = get_image_files(path,file_extension)
	image_files.sort()
	try:
		i = 0
		#get dimension of the image. It assumes all images are the same resolution
		first_img = Image.open(path + "/" + image_files[0])
		size_of_images = first_img.size
		
		m = np.zeros((len(image_files),size_of_images[0],size_of_images[1]), dtype = int)
		#m = m.astype(int)
		for file in image_files:
			print "Opening image at " + path + "/" + file
			img = Image.open(path + "/" + file)
			a = img_to_array(img)
			m[i] = a
			i += 1
		#print m	
	except IOError:
		print "Problems opening files from" + path

import warnings
 
def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1,ploton=False):
		"""
		Anisotropic diffusion.
 
		Usage:
		imgout = anisodiff(im, niter, kappa, gamma, option)
 
		Arguments:
				img	- input image
				niter  - number of iterations
				kappa  - conduction coefficient 20-100 ?
				gamma  - max value of .25 for stability
				step   - tuple, the distance between adjacent pixels in (y,x)
				option - 1 Perona Malik diffusion equation No 1
						 2 Perona Malik diffusion equation No 2
				ploton - if True, the image will be plotted on every iteration
 
		Returns:
				imgout   - diffused image.
 
		kappa controls conduction as a function of gradient.  If kappa is low
		small intensity gradients are able to block conduction and hence diffusion
		across step edges.  A large value reduces the influence of intensity
		gradients on conduction.
 
		gamma controls speed of diffusion (you usually want it at a maximum of
		0.25)
 
		step is used to scale the gradients in case the spacing between adjacent
		pixels differs in the x and y axes
 
		Diffusion equation 1 favours high contrast edges over low contrast ones.
		Diffusion equation 2 favours wide regions over smaller ones.
 
		Reference:
		P. Perona and J. Malik.
		Scale-space and edge detection using ansotropic diffusion.
		IEEE Transactions on Pattern Analysis and Machine Intelligence,
		12(7):629-639, July 1990.
 
		Original MATLAB code by Peter Kovesi  
		School of Computer Science & Software Engineering
		The University of Western Australia
		pk @ csse uwa edu au
		<http://www.csse.uwa.edu.au>
 
		Translated to Python and optimised by Alistair Muldal
		Department of Pharmacology
		University of Oxford
		<alistair.muldal@pharm.ox.ac.uk>
 
		June 2000  original version.	  
		March 2002 corrected diffusion eqn No 2.
		July 2012 translated to Python
		"""
 
		# ...you could always diffuse each color channel independently if you
		# really want
		
		if img.ndim == 3:
				warnings.warn("Only grayscale images allowed, converting to 2D matrix")
				img = img.mean(2)
 
		# initialize output array
		img = img.astype('float32')
		imgout = img.copy()
 
		# initialize some internal variables
		deltaS = np.zeros_like(imgout)
		deltaE = deltaS.copy()
		NS = deltaS.copy()
		EW = deltaS.copy()
		gS = np.ones_like(imgout)
		gE = gS.copy()
 
		# create the plot figure, if requested
		if ploton:
				import pylab as pl
				from time import sleep
 
				fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
				ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
 
				ax1.imshow(img,interpolation='nearest')
				ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
				ax1.set_title("Original image")
				ax2.set_title("Iteration 0")
 
				fig.canvas.draw()
 
		for ii in xrange(niter):
 
				# calculate the diffs
				deltaS[:-1,: ] = np.diff(imgout,axis=0)
				deltaE[: ,:-1] = np.diff(imgout,axis=1)
 
				# conduction gradients (only need to compute one per dim!)
				if option == 1:
						gS = np.exp(-(deltaS/kappa)**2.)/step[0]
						gE = np.exp(-(deltaE/kappa)**2.)/step[1]
				elif option == 2:
						gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
						gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
 
				# update matrices
				E = gE*deltaE
				S = gS*deltaS
 
				# subtract a copy that has been shifted 'North/West' by one
				# pixel. don't as questions. just do it. trust me.
				NS[:] = S
				EW[:] = E
				NS[1:,:] -= S[:-1,:]
				EW[:,1:] -= E[:,:-1]
 
				# update the image
				imgout += gamma*(NS+EW)
 
				if ploton:
						iterstring = "Iteration %i" %(ii+1)
						ih.set_data(imgout)
						ax2.set_title(iterstring)
						fig.canvas.draw()
						#sleep(0.01)
						#print images at each step 
						#ani_img = Image.fromarray(np.uint8(imgout)) 
						#ani_img.save("ani_img" + str(ii) + ".png")
						
 
		return imgout
 
def anisodiff3(stack,niter=1,kappa=50,gamma=0.1,step=(1.,1.,1.),option=1,ploton=False):
		"""
		3D Anisotropic diffusion.
 
		Usage:
		stackout = anisodiff(stack, niter, kappa, gamma, option)
 
		Arguments:
				stack  - input stack
				niter  - number of iterations
				kappa  - conduction coefficient 20-100 ?
				gamma  - max value of .25 for stability
				step   - tuple, the distance between adjacent pixels in (z,y,x)
				option - 1 Perona Malik diffusion equation No 1
						 2 Perona Malik diffusion equation No 2
				ploton - if True, the middle z-plane will be plotted on every
						 iteration
 
		Returns:
				stackout   - diffused stack.
 
		kappa controls conduction as a function of gradient.  If kappa is low
		small intensity gradients are able to block conduction and hence diffusion
		across step edges.  A large value reduces the influence of intensity
		gradients on conduction.
 
		gamma controls speed of diffusion (you usually want it at a maximum of
		0.25)
 
		step is used to scale the gradients in case the spacing between adjacent
		pixels differs in the x,y and/or z axes
 
		Diffusion equation 1 favours high contrast edges over low contrast ones.
		Diffusion equation 2 favours wide regions over smaller ones.
 
		Reference:
		P. Perona and J. Malik.
		Scale-space and edge detection using ansotropic diffusion.
		IEEE Transactions on Pattern Analysis and Machine Intelligence,
		12(7):629-639, July 1990.
 
		Original MATLAB code by Peter Kovesi  
		School of Computer Science & Software Engineering
		The University of Western Australia
		pk @ csse uwa edu au
		<http://www.csse.uwa.edu.au>
 
		Translated to Python and optimised by Alistair Muldal
		Department of Pharmacology
		University of Oxford
		<alistair.muldal@pharm.ox.ac.uk>
 
		June 2000  original version.	  
		March 2002 corrected diffusion eqn No 2.
		July 2012 translated to Python
		"""
 
		# ...you could always diffuse each color channel independently if you
		# really want
		if stack.ndim == 4:
				warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
				stack = stack.mean(3)
 
		# initialize output array
		stack = stack.astype('float32')
		stackout = stack.copy()
 
		# initialize some internal variables
		deltaS = np.zeros_like(stackout)
		deltaE = deltaS.copy()
		deltaD = deltaS.copy()
		NS = deltaS.copy()
		EW = deltaS.copy()
		UD = deltaS.copy()
		gS = np.ones_like(stackout)
		gE = gS.copy()
		gD = gS.copy()
 
		# create the plot figure, if requested
		if ploton:
				import pylab as pl
				from time import sleep
 
				showplane = stack.shape[0]//2
 
				fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
				ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
 
				ax1.imshow(stack[showplane,...].squeeze(),interpolation='nearest')
				ih = ax2.imshow(stackout[showplane,...].squeeze(),interpolation='nearest',animated=True)
				ax1.set_title("Original stack (Z = %i)" %showplane)
				ax2.set_title("Iteration 0")
 
				fig.canvas.draw()
 
		for ii in xrange(niter):
 
				# calculate the diffs
				deltaD[:-1,: ,:  ] = np.diff(stackout,axis=0)
				deltaS[:  ,:-1,: ] = np.diff(stackout,axis=1)
				deltaE[:  ,: ,:-1] = np.diff(stackout,axis=2)
 
				# conduction gradients (only need to compute one per dim!)
				if option == 1:
						gD = np.exp(-(deltaD/kappa)**2.)/step[0]
						gS = np.exp(-(deltaS/kappa)**2.)/step[1]
						gE = np.exp(-(deltaE/kappa)**2.)/step[2]
				elif option == 2:
						gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
						gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
						gE = 1./(1.+(deltaE/kappa)**2.)/step[2]
 
				# update matrices
				D = gD*deltaD
				E = gE*deltaE
				S = gS*deltaS
 
				# subtract a copy that has been shifted 'Up/North/West' by one
				# pixel. don't as questions. just do it. trust me.
				UD[:] = D
				NS[:] = S
				EW[:] = E
				UD[1:,: ,: ] -= D[:-1,:  ,:  ]
				NS[: ,1:,: ] -= S[:  ,:-1,:  ]
				EW[: ,: ,1:] -= E[:  ,:  ,:-1]
 
				# update the image
				stackout += gamma*(UD+NS+EW)
 
				if ploton:
						iterstring = "Iteration %i" %(ii+1)
						ih.set_data(stackout[showplane,...].squeeze())
						ax2.set_title(iterstring)
						fig.canvas.draw()
						# sleep(0.01)
					
		return stackout

		
#read_from_file_to_array(path,file_extension)


#trying the median filter
#original_img = Image.open("spray2.png")
#size = original_img.size
img = Image.open('fruits_noisy.jpg')
img_array = np.asarray(img)
#img = Image.fromarray(np.uint8(img_array))


#median_filter(input, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
#due to some problems with defining the footprint/size, currently it is assumed that all images are square NxN

filtered_array = im.median_filter(img_array, 5 , None , None, 'wrap', 0.0, 0)
filtered_img = Image.fromarray(np.uint8(filtered_array)) 

filtered_img.save("filtered_img.png")

print "anisodiff" 
anisodiff(img_array,50,50,0.1,(1.,1.),1,True)

