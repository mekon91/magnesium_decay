
import os,sys, scipy, warnings, time, helpFunctions, convert
import numpy as np
from os import listdir
from os.path import isfile, join
import scipy.ndimage as im
from PIL import Image, ImageSequence
from images2gif import writeGif
import matplotlib.pyplot as plt

file_extension = ".tif"

#dictionary of colors, 1 green (decaying magnesium), 2 dark blue (pure magnesium)				
color_dictionary = { 1 : (0,255,0), 2 : (0,0,125) }

class processFolder:
	#paths
	path = None
	path_original_images = None
	path_original_sli = None
	path_filtered = None
	path_recolored_basic = None
	# 3D numpy array holding the data from all images. m[i] is image i 
	m = []
	# 3D numpy array holding the region to which each entry is classified. 
	# In the magnesium_decay, it should have 0,1,2 for air, decaying mag, pure mag 
	m_regions = []
	
	
	def __init__(self, path, range_files, threshholds): 
		
		self.range_files = range_files
		self.threshholds = threshholds
		self.path  = path
		
		path_original_images = path + "/tiff"
		self.path_original_images = path_original_images
		path_original_sli = path + "/reco"
		self.path_original_sli = path_original_sli
		
		
		path_mask = path + "/Masked"
		if not os.path.exists(path_mask):
			os.makedirs(path_mask)
		
		
		path_filtered = path + "/Filtered"
		if not os.path.exists(path_filtered): 
			os.makedirs(path_filtered)
		self.path_filtered = path_filtered		
			
		path_recolored_basic = path + "/Recolored"	
		if not os.path.exists(path_recolored_basic):
			os.makedirs(path_recolored_basic)
		self.path_recolored_basic = path_recolored_basic	
			
		#m is the 3D array containing all the images at path	
		m = convert.images_to_numpy3D(path_original_images, range_files, file_extension)
		m = apply_filters(m, path_filtered)
		self.m = m	
				
		print "Number of images: " + str(m.shape[0])		
			
		r = region_array(m,self.threshholds)
		self.m_regions = r	
		save_images_from_numpy3D(r, 0, path_mask)
		
		save_images_from_numpy3D_and_dictionary(r, color_dictionary, path_recolored_basic)	
		
																		
def make_video(filename, path_images, path_save):
	images = get_PIL_images(path_images)
	writeGif(path_save + "/" +filename, images, duration = 0.015)
	print	

#takes numpy3D, writes sobel filtered images to path
def apply_sobel(m):
	n = np.zeros(m.shape)
	for i in xrange(m.shape[0]):
		print "Applying sobel filter: " + str(i) + "/" + str(m.shape[0])
		a = m[i]
		dx = im.sobel(a, 0)  # horizontal derivative
		dy = im.sobel(a, 1)  # vertical derivative
		mag = np.hypot(dx, dy)  # magnitude
		mag *= 255.0 / np.max(mag)  # normalize (Q&D)
		n[i] = (mag)
	return n				

#takes numpy3D m and returns an array n that 
def region_of_interest(m,(minimum,maximum)):
	n = np.zeros(m.shape)
	for i in xrange(m.shape[0]):
		print "Finding region of interest: " + str(i) + "/" + str(m.shape[0])
		for j in xrange(m.shape[1]):
			for k in xrange(m.shape[2]):
				if (minimum < m[i,j,k] and m[i,j,k] <= maximum):
					n[i,j,k] = 1
				else:
					n[i,j,k] = 0	 
	return n

#returns array of indices for the region of interest. A point in the region of interest has value between min and max
def region_of_interest_indices(m, (minimum,maximum)):
	indices = []
	for i in xrange(m.shape[0]):
		print "Finding region of interest indices: " + str(i) + "/" + str(m.shape[0])
		for j in xrange(m.shape[1]):
			for k in xrange(m.shape[2]):
				if ( minimum < m[i,j,k] <= maximum ):
					indices.append( (i,j,k) ) 
	return indices		
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
 
def anisodiff3(stack,niter=1,kappa=50,gamma=0.1,step=(1.,1.,1.),option=1):
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
				print str(ii) + "/" + str(niter) + " iterations of the anisotropic diffusion done"
					
		return stackout

#takes a filtered 3D array and the threshholds for different images. Sets 0, 1 ,2 ... for corresponding regions 
def region_array(m,threshholds):
	m_regions = np.zeros(m.shape)
	print "Converting array of size " + str(m.shape) + "to a region array..." 
	for i in xrange(m.shape[0]):
		print "Converted slice: " + str(i) + "/" + str(m.shape[0]) 
		for j in xrange(m.shape[1]):
			for k in xrange(m.shape[2]):
				for t in xrange(0,len(threshholds)):
					if (m[i,j,k] > threshholds[t]):
						m_regions[i,j,k] = t + 1			
	return m_regions
	
			
#returns counts of  decaying and pure. No optimisation.
def get_counts(m):
	decaying = 0
	pure = 0 
	for i in xrange(m.shape[0]):
		print "Getting counts " + str(i) + "/" + str(m.shape[0])
		for j in xrange(m.shape[1]):
			for k in xrange(m.shape[2]):
				if m[i,j,k] == 1 :
					decaying += 1
				if m[i,j,k] == 2 :
					pure += 1	
	print "\n DECAYING = " + str(decaying) + " PURE = " + str(pure)					
	return (decaying, pure) 	

#applies filters on the 3D array M, saves saves the images (horizontal slices) at path_filtered
def apply_filters(m, path_filtered):
	
	start = time.clock()

	print "Applying anisotropic diffusion on 3D array of size: " + str(m.shape)  
	m = anisodiff3(m,niter=15,kappa=8,gamma=0.1,step=(1.,1.,1.),option=1)
	print "Done in: " + str(time.clock() - start) + " seconds"
	print
	
	start = time.clock()
	
	median_footprint = 7
	print "Applying median filter with footprint of " + str(median_footprint) + " on 3D array of size: " + str(m.shape) 
	m = im.median_filter(m, median_footprint , None , None, 'reflect', 0.0, 0)
	print "Done in: " + str(time.clock() - start) + " seconds"
	
	#loop over all images in m and save them
	for i in xrange(m.shape[0]):
		img = Image.fromarray(np.uint8(m[i]))
		img.save(path_filtered + "/filtered_img" + str(i) + ".tif")
		print "Saving image " + "/filtered_img" + str(i) + ".tif" + " at " + path_filtered + "/filtered_img" + str(i) + ".tif"
	
	return m	

#colors the images and saves the recolored versions at the path_recolored. Advanced recoloring
def recolor2(path_images, path_recolored):
	i = 0
	print 'Recoloring images at path: ' + path_recolored + ' ...\n' 
	images = get_PIL_images(path_images)
	start = time.clock()
	for img in images:
		out_img = Image.new("RGB", img.size, "black")
		for x in xrange(img.size[0]):
			for y in xrange(img.size[1]):				
				gray = img.getpixel((x,y))
					
				#some threshholds. TODO: Should be read from a file! 
				t1 = 11.
				t2 = 25.
				t3 = 50.
				t4 = 185.
				t5 = 255.
				#TODO: READ FROM FILES!
				#some colors used 
				pink2 = (136,0,136) 
			    
				#try linear interpolation
				darkblue = (0, 0, 128)
				lightblue = (30, 144, 255)

				darkgreen = (3, 40, 3)
				lightgreen = (127, 255, 0)

				darkred = (94, 38, 18)
				lightred = (238, 99, 99)

				darkpurple =   (128, 0, 128)
				lightpurple =  (238, 0, 238)
				
				#vacuum
				if (gray <= t1):
					out_img.putpixel( (x,y), black)
				#air	
				elif (gray <= t2):
					normalizer = (gray-t1)/(t2-t1)
					out_img.putpixel( (x,y), tuple([int(z * normalizer) for z in pink2]) )
				#dense air	
				elif (gray <= t3):
					d = t3 - t2
					norm1 = (gray - t2) / d
					norm2 = (t3 - gray) / d
					out_img.putpixel( (x,y), tuple([int((norm1*a) + (norm2*b)) for a, b in zip(lightpurple, darkpurple)]))									
				#decaying magnesium	
				elif (gray <= t4):
					#linear interpolation from lightgreen -> darkgreen
					d = t4 - t3
					norm1 = (gray - t3) / d
					norm2 = (t4 - gray) / d
					out_img.putpixel( (x,y), tuple([int((norm1*a) + (norm2*b)) for a, b in zip(lightgreen, darkgreen)]))					
					#out_img.putpixel( (x,y), lawngreen)
					#pure magnesium	
				else:
					#linear interpolation from lightblue -> darkblue
					d = t5 - t4
					norm1 = (gray - t4) / d
					norm2 = (t5 - gray) / d
					out_img.putpixel( (x,y), tuple([int((norm1*a) + (norm2*b)) for a, b in zip(lightblue, darkblue)]))	
										
					#out_img.putpixel( (x,y), blue4)
				
		print "Writing image recolored" + str(i) + ".tif at " + path_recolored + "/recolored" + str(i) + ".tif"
		print
									
		out_img.save(path_recolored + '/recolored' + str(i) + '.tif')
		i += 1
	print "Done writing recolored images. It took " + str(time.clock() - start) + " seconds "
			
#gets vertical slices along a dimension from the 3D array m, along the the axis dimension and saves them at path.
def save_images_from_numpy3D (m, dimension, vertical_path):
	
	if (dimension <= 2 and dimension >= 0):
		
		
		if (dimension == 0):
			print "Saving slice images along the i'th (0) dimension"
			for i in xrange(m.shape[dimension]):
				img = Image.fromarray(np.uint8(m[i,:,:])) 
				img.save(vertical_path + "/slice_i_" + str(i) + ".tif")
				print "Writing image " + "/slice_i_" + str(i) + ".tif" + " at " + vertical_path + "/slice_i_" + str(i) + ".tif"
		elif (dimension == 1): 
			print "Saving slice images along the j'th (1) dimension"
			for i in xrange(m.shape[dimension]):			
				img = Image.fromarray(np.uint8(m[:,i,:]))
				img.save(vertical_path + "/slice_j_" + str(i) + ".tif")
				print "Writing image " + "/slice_j_" + str(i) + ".tif" + " at " + vertical_path + "/slice_j_" + str(i) + ".tif"
		else: 		
			print "Saving slice images along the k'th (1) dimension"
			for i in xrange(m.shape[dimension]):
				img = Image.fromarray(np.uint8(m[:,:,i]))
				img.save(vertical_path + "/slice_k_" + str(i) + ".tif")	
				print "Writing image " + "/slice_k_" + str(i) + ".tif" + " at " + vertical_path + "/slice_k_" + str(i) + ".tif"			
	else:
		print "Dimension passed to get_PIL_images_from_3Darray must be among [0,1,2]"


def save_images_from_numpy3D_and_dictionary(m, color_dictionary, path):
	for i in xrange(m.shape[0]):
		m[i] = m[i]
		img = Image.new("RGB", (775,775), "black")
		for x in xrange(775):
			for y in xrange(775):
				if (m[i,x,y] != 0):
					img.putpixel( (y,x), color_dictionary[m[i,x,y]] )
		img.save(path + "/recolored" + str(i) + ".tif")
		print "Saved image " + 	"/recolored" + str(i) + ".tif" + " at " + path + "/recolored" + str(i) + ".tif"

#might do this more complex, like save_PIL_iomages_from_3Darray
def save_images_from_3D_region_array (m_regions, path_basic_recolor):
	#Colors will probably be one of the parameters of this function. For now just green and blue
	color_dict = { 1 : (0,255,0), 2 : (0,0,125) } 
	#loop over all horisontal slices of m (images)
	for i in xrange(m_regions.shape[0]):
		img = Image.new("RGB", (775,775), "black")	
		for x in xrange(775):
			for y in xrange(775):
				if (m_regions[i,x,y] != 0):
					img.putpixel( (x,y), color_dict[1])   
		img.save(path_basic_recolor + "/img" + str(i) + ".tif")
		print "Saved image " + 	"/img" + str(i) + ".tif" + " at " + path_basic_recolor + "/img" + str(i) + ".tif"

#for now takes 2 region 3Dnumpy arrays. Assumes r1 is from earlier data than r2. 
def region_of_interest_time_evolution(r1, r2):
	if (r1.shape != r2.shape):
		print "Regions passed to region_of_interest_time_evolutions should have same sizes"
		return None
	
	#declare returned region
	r = np.zeros(r1.shape)
	
	for i in xrange(r1.shape[0]):
		print "Region of interest time evolution: " + str(i) + "/" + str(r.shape[0])
		for j in xrange(r1.shape[1]):
			for k in xrange(r1.shape[2]):
				#decaying to pure, should not happen!
				if (r1[i,j,k] == 1 and r2[i,j,k] == 2):
					r[i,j,k] = 4   #6	
				if (r1[i,j,k] == 0 and r2[i,j,k] == 1):
					r[i,j,k] = 6		
					
				elif (r1[i,j,k] == 1 and r2[i,j,k] == 1):
					r[i,j,k] = 1
				elif (r1[i,j,k] == 2 and r2[i,j,k] == 2):
					r[i,j,k] = 2		
									
				#decaying magnesium -> air
				elif (r1[i,j,k] == 1 and r2[i,j,k] == 0):
					r[i,j,k] = 5
				#pure magnesium -> decaying magnesium
				elif (r1[i,j,k] == 2	and r2[i,j,k] == 1):
					r[i,j,k] = 4	
				#fail, nothing to magnesium. Shows where alignment is not right	
				elif (r1[i,j,k] == 0 and (r2[i,j,k] == 1 or r2[i,j,k] == 2)):
					r[i,j,k] = 5
				elif (r1[i,j,k] == 0 and (r2[i,j,k] == 0)):
					r[i,j,k] = 0
				else:
					r[i,j,k] = 6				
	return r					

		
	
#should align m2 with m1. The 2 arrays should be region arrays with 1 for any magnesium and 0 for air
#It should return a 3d vector (x,y) stating how much m2 should be shifted to be aligned with m1
def align(m1,m2):	
	reached_max = False
	#vector will record how much to shift m2 to get as close as possible to m1
	vector = [0,0]
	#shift the m2 in 4 directions in the j,k plane (i.e. in the images plane)
	while (not reached_max):
		
		#equivalent of dot product square rooted. This works faster than pythons dot product
		initial_similarity = np.linalg.norm(np.multiply(m1,m2))
			
		#similarities with the neighbor positions
		north_similarity  = np.linalg.norm(np.multiply(m1,np.roll(m2, -1, axis = 1)))
		south_similarity = np.linalg.norm(np.multiply(m1,np.roll(m2,  1, axis = 1)))
		east_similarity = np.linalg.norm(np.multiply(m1,np.roll(m2, 1 , axis = 2)))
		west_similarity = np.linalg.norm(np.multiply(m1,np.roll(m2, -1 ,axis = 2)))	
		similarities = (north_similarity, south_similarity, east_similarity, west_similarity, initial_similarity)
		maximum = max(similarities) 
		print
		print similarities
		print
		if (north_similarity == maximum) :
			print "Shifting north"
			print
			vector[0] -= 1
			m2 = np.roll(m2, -1, axis = 1)
		elif (south_similarity == maximum):
			print "Shifting south"
			vector[0] += 1 
			m2 = np.roll(m2,  1, axis = 1)
		elif (east_similarity == maximum):
			print "Shifting east"
			vector[1] += 1
			m2 = np.roll(m2, 1 , axis = 2)
		elif (west_similarity == maximum):
			print "Shifting west"
			vector[1] -= 1
			m2 = np.roll(m2, -1 ,axis = 2)
		else:
			reached_max = True			
	return vector			 






