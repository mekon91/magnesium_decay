
import Image,os,sys, scipy, warnings, time, cv
import numpy as np
from os import listdir
from os.path import isfile, join
import scipy.ndimage as im





file_extension = '.tif'
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
		return None
	image_files.sort()	
	return image_files	


#for now this does not work well, i hate iplimage	
def make_video(filename,images):
	frame_size = (images[0].width,images[0].height)  
	#cv.CV_FOURCC('i','Y', 'U', 'V')
	writer = cv.CreateVideoWriter(filename, 0, 25, frame_size, is_color = cv.CV_LOAD_IMAGE_GRAYSCALE)
	for img in images:
		cv.WriteFrame(writer,img)
	

#using the get_image_files, it then reads all images at path with the extension file_extension, and stacks them on a 3D array m. m[i] is the ith image
def read_from_file_to_array(path,file_extension):
	image_files = get_image_files(path,file_extension)
	try:
		i = 0
		#get dimension of the image. It assumes all images are the same resolution
		first_img = Image.open(path + "/" + image_files[0])
			
		m = np.zeros((len(image_files),first_img.size[0],first_img.size[1]), dtype = int)
			
		for file in image_files:
			print "Opening image at " + path + "/" + file
			
			img = Image.open(path + "/" + file)			
			#img = cv.LoadImage(path + "/" + file, cv.CV_LOAD_IMAGE_GRAYSCALE)				
			#convert the image to a 2D array.
			a = np.asarray(img)		
			m[i] = a
			#if (i == 60):	
			#	lol_array = m[:,:,350]
			#	lol_image = Image.fromarray(np.uint8(lol_array))
			#	lol_image.show()
			#	img.show()
			i += 1	
	except IOError:
		print "Problems opening files from" + path	
	return m
	
def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1):
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
				# pixel. 
				NS[:] = S
				EW[:] = E
				NS[1:,:] -= S[:-1,:]
				EW[:,1:] -= E[:,:-1]
 
				# update the image
				imgout += gamma*(NS+EW)		
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
 
		step is used to scale the gradients in case the spacing between adjacent
		pixels differs in the x,y and/or z axes
 
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


img = Image.open("mag_test.tif")
#for some reason the x axis is flipped when reading images, thus flipud was used to flip back to original			
img_array = np.flipud(np.asarray(img))

#median filter used, 
median_array = im.median_filter(img_array, 5 , None , None, 'wrap', 0.0, 0)
median_img = Image.fromarray(np.uint8(median_array))
median_img.save("median_img.tif")


ani_array = anisodiff(img_array,10,50,0.1,(1.,1.),1)
ani_img = Image.fromarray(np.uint8(ani_array))
ani_img.save("ani_img.tif")

median_img.show()
ani_img.show()





