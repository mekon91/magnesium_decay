
import os,sys, scipy, warnings, time, helpFcts, fileReader 
import numpy as np
from os import listdir
from os.path import isfile, join
import scipy.ndimage as im
from PIL import Image, ImageSequence
from images2gif import writeGif
from regionGrowing import *
import matplotlib.pyplot as plt

file_extension = ".tif"
path = "/afs/desy.de/user/d/dariep/Desktop/magnesium_decay/Data1"

class processImages:
	#path of the images
	path = None
	# 3D numpy array holding the data from all images. m[i] is image i 
	m = []
	# 3D numpy array holding the region to which each entry is classified. 
	# In the magnesium_decay, it should have 0,1,2 for air, decaying mag, pure mag 
	m_regions = []
	
	
	def __init__(self, path): 
		
		self.path  = path
		
			
		path_original_images = path + "/tiff"
		path_original_sli = path + "/reco"
		
		#creating new paths: 1 for filtered images, 1 for recolored images, 1 for gif videos, 
		#plus 4 for slices along ith and jth dimensions (2 filtered, 2 recolored)	
		
		path_filtered = path + "/Filtered"
		if not os.path.exists(path_filtered): 
			os.makedirs(path_filtered)
			
		path_recolored = path + "/Nicely_Recolored"
		if not os.path.exists(path_recolored):
			os.makedirs(path_recolored)
		
		#slices along the j'th (1) dimension
		path_slice_j = path + "/Slice_j"
		if not os.path.exists(path_slice_j):
			os.makedirs(path_slice_j)
			
		#slices along the k'th (2) dimension
		path_slice_k = path + "/Slice_k"
		if not os.path.exists(path_slice_k):
			os.makedirs(path_slice_k)
				
		#recolored slices along the j'th(2) dimension
		path_slice_j_recolored = path + "/Slice_j_Recolored"
		if not os.path.exists(path_slice_j_recolored):
			os.makedirs(path_slice_j_recolored)
			
		#recolored slices along the k'th(2) dimension
		path_slice_k_recolored = path + "/Slice_k_Recolored"
		if not os.path.exists(path_slice_k_recolored):
			os.makedirs(path_slice_k_recolored)		
			
		path_gifs = path + "/GIFs"	
		if not os.path.exists(path_gifs):
			os.makedirs(path_gifs)
		
		
		path_recolored_basic = path + "/Recolored"	
		if not os.path.exists(path_recolored_basic):
			os.makedirs(path_recolored_basic)
			
		#m is the 3D array containing all the images at path
		m = fileReader.image_from_folder_to_array(path_original_images, file_extension)
		self.m = m
		
		f = frequencies(m)
	
		print "Number of images: " + str(m.shape[0])		
			
		#m = apply_filters(m, path_filtered)
		#region grow to find the different regions.Should return list of thresholds.TODO
		#threshholds = [50,185]
			
		#self.m_regions = region_array(m,threshholds)
		
		#save_images_from_3D_region_array (self.m_regions, path_recolored_basic)
		
				
		#need to make a BASIC recolor function, one color per region, no interpolation or anything cool
		#recolor2(path_filtered, path_recolored)
		#recolor(path_filtered, path_recolored_basic)
						
		#save_PIL_images_from_3Darray(m, 1, path_slice_j)
		#save_PIL_images_from_3Darray(m, 2, path_slice_k)
		
		#over writing for now
		#recolor(path_slice_j, path_slice_j_recolored)
		#recolor(path_slice_k, path_slice_k_recolored)
				
		#make_video("filtered.GIF", path_filtered, path_gifs) 
		#make_video("recolored.GIF", path_recolored, path_gifs) 
		#make_video("slices_j.GIF", path_slice_j, path_gifs) 
		#make_video("slices_k.GIF", path_slice_k_recolored, path_gifs) 				


def make_video(filename, path_images, path_save):
	images = get_PIL_images(path_images)
	writeGif(path_save + "/" +filename, images, duration = 0.015)
	print	
	
def get_PIL_images(path):
	image_names = get_image_files(path)
	images = []
	for img_name in image_names:
		img = Image.open(path + "/" + img_name)
		images.append(img)
	return images
	
#returns an array of size 256, each entry shows the frequency of the corresponding grayscale valye
#FOR NOW ITS COUNTS NOT FREQUENCY. TODO
def frequencies(m):
	f = np.zeros(256,dtype = int)
	for i in xrange(m.shape[0]):
		print str(i)
		for j in xrange(m.shape[1]):
			for k in xrange(m.shape[2]):
				f[m[i,j,k]] += 1
	file_freq = open('frequencies.txt','w')
	for i in f:
		
		file_freq.write(str(i) + '\n')
	file_freq.close()	
	return f 
	
def frequencies_float(m):
	f = np.zeros(512,dtype = int)
	#get max and min
	minimum = m[0,0,0]
	maximum = m[0,0,0]
	print m[0,0,0]
	 
	"""for i in xrange(m.shape[0]):
		print "Determining max and min " + str(i) + "% "
		for j in xrange(m.shape[1]):
			for k in xrange(m.shape[2]):
				if (m[i,j,k] < minimum):
					minimum = m[i,j,k]				
				if (m[i,j,k] > maximum):
					maximum = m[i,j,k]
	print "minimum is " + str(minimum) + "\n maximum is " + str(maximum)
	"""
	maximum = 0.0153947146609 
	minimum = -0.00610217032954
	
	bin_size = (maximum - minimum) / 512.
	print "Bin size is " + str(bin_size)
	for i in xrange(m.shape[0]):
		print "Float frequencies " + str(i) + "% "
		for j in xrange(m.shape[1]):
			for k in xrange(m.shape[2]):
				index = abs(int(m[i,j,k] / bin_size))
				f[index] += 1 
	return f
					
#reads image from 'path', writes an image to 'filename'. 
def sobel_edge_detection(path,filename):
	img2 = scipy.misc.imread(path)
	img2 = img2.astype('int32')
	img2_array = np.asarray(img2)
	dx = ndimage.sobel(img2_array, 0)  # horizontal derivative
	dy = ndimage.sobel(img2_array, 1)  # vertical derivative
	mag = np.hypot(dx, dy)  # magnitude
	mag *= 255.0 / np.max(mag)  # normalize (Q&D)
	scipy.misc.imsave(filename, mag)
	
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

#takes a filtered 3D array and the threshholds for different images
def region_array(m,threshholds):
	if (len(m.shape) != 3):
		print "The array passed to region_array is not a 3D array!"
		return None
	else:
		m_regions = np.zeros(m.shape)
		print "Converting array of size " + str(m.shape) + "to a region array..." 
		for i in xrange(m.shape[0]):
			print "Converted slice: " + str(i) 
			for j in xrange(m.shape[1]):
				for k in xrange(m.shape[2]):
					for t in range(1,len(threshholds)):
						if (m[i,j,k] > threshholds[t]):
							#This will overwrite values. Is it good method? TODO !!!!
							m_regions[i,j,k] = t   
		return m_regions

def test_script():
	#read_from_file_to_array(path,file_extension)
	img = Image.open('mag_test.tif')
	#for some reason the x axis is flipped when reading images, thus flipud was used to flip back to original			
	img_array = np.flipud(np.asarray(img))

	#median filter used, 
	median_array = im.median_filter(img_array, 8 , None , None, 'reflect', 0.0, 0)
	median_img = Image.fromarray(np.uint8(median_array))
	median_img.save('median_img.tif')


	ani_array = anisodiff(img_array,niter=30,kappa=8,gamma=0.1,step=(1.,1.),option=1)
	ani_array = im.median_filter(ani_array, 7 , None , None, 'reflect', 0.0, 0)
	
	ani_img = Image.fromarray(np.uint8(ani_array))
	ani_img.save('ani_img.tif')
	#edge detection test
	#sobel_edge_detection('ani_img.tif','sobel.tif')

	#distance transform test
	#dtrans_img = Image.open('ani_img.tif')
	#dtrans_array = np.array(dtrans_img)
	#dtrans_array = im.distance_transform_bf(dtrans_array, metric='euclidean', sampling=None, return_distances=True, return_indices=False, distances=None, indices=None)
	#dtrans_img = Image.fromarray(np.uint8(dtrans_array))
	
	#gradient test
	#grad_array  = im.gaussian_gradient_magnitude(median_array, (1,1) , output=None, mode='reflect', cval=0.0)
	#grad_img = Image.fromarray(np.uint8(grad_array))
	#grad_img.save('grad_img.tif')
	print
	
#test method to color air -> black, "air filaments" -> pink, affected magnesium -> red, pure magnesium -> blue	
#def recolor(images):
def test_ani():
	m = read_from_file_to_array(path,file_extension)
	print "Number of images: " + str(m.shape[0])

	start = time.clock()

	m = anisodiff3(m,niter=15,kappa=8,gamma=0.1,step=(1.,1.,1.),option=1)	
	m = im.median_filter(m, 7 , None , None, 'reflect', 0.0, 0)
	for i in xrange(m.shape[0]):
		#file index. Needs to be 01, 02 ... 09, 10  . not 1, 2 ... 9, 10 
		if (i <= 9):
			index = '0' + str(i)
		else:
			index = str(i)	
		img = Image.fromarray(np.uint8(m[i]))
		img.save(save_path + "/img" + index + ".tif")

	end = time.clock()
	print "Time for anisotropic diffusion + median filter on " + str(m.shape[0]) + " images is " + str(end - start)

#applies filters on the 3d array M, saves saves the images (horizontal slices) at path_filtered
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

#colors the images and saves the recolored versions at the path_recolored.
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
			
def test_recolor(image_files,path,recolored_path):
	
	i = 0
	for img_name in image_files:
		img = Image.open(path + "/" + img_name)
		print "Reading image " + img_name + " at " + path + "/" + img_name
		out_img = Image.new("RGB", img.size, "black")
		for x in xrange(img.size[0]):
			for y in xrange(img.size[1]):				
				gray = img.getpixel((x,y))
				#r,g,b = out_img.getpixel((x,y))
				#print("Red: {0}, Green: {1}, Blue: {2}".format(r,g,b))
				#pink 2: 238 	169 	184
				
				#some threshholds. TODO: Should be read from a file! 
				t1 = 11.
				t2 = 25.
				t3 = 50.
				t4 = 185.
				t5 = 255.
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
					#normalizer = (gray-t2)/(t3-t2)
					#out_img.putpixel( (x,y), tuple([int(z * normalizer) for z in mint]) )
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
		#file index. 
		if (i <= 9):
			index = '0' + str(i)
		else:
			index = str(i)				
		print "Writing image recolored" + index + ".tif at " + recolored_path 
		print							
		out_img.save(recolored_path + '/recolored' + index + '.tif')
		i += 1		

def test_vertical_slice(image_files,path, vertical_path):
	m = read_from_file_to_array(path,file_extension)
	print "Number of images: " + str(m.shape[1])
	
	start = time.clock()
	
	for i in xrange(m.shape[1]):
		img = Image.fromarray(np.uint8(m[:,:,i]))
		img.save(vertical_path + "/img" + str(i) + ".tif")
		print "Writing vertical slice image img" + str(i) + ".tif at " + vertical_path
		print	
	end = time.clock()
	
	
	print "Time to write " + str(m.shape[2]) + " images is " + str(end-start)

#gets vertical slices along a dimension from the 3D array m, along the the axis dimension and saves them at path. Returns the images
def save_PIL_images_from_3Darray (m, dimension, vertical_path):
	
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

# might do this more complex, like save_PIL_iomages_from_3Darray
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
		print "Saved image " + 	"/recolored_img" + str(i) + ".tif" + " at " + path_basic_recolor + "/img" + str(i) + ".tif"

def histogram(filename):
	try:
		file_freq = open(filename,'r')
		nr_lines = 0
		freq_list = []
		for line in file_freq:			
			nr_lines += 1			
			freq_list.append(int(line))
		f = np.array( freq_list ) 
		i = 0
					
		alphab = range(0,nr_lines)	
		print str(len(alphab)) + " " + str(len(f))
				
		pos = np.arange(len(alphab))
		width = 0.7    # gives histogram aspect to the bar diagram

		ax = plt.axes()
		ax.set_xticks(pos + (width / 2))
		ax.set_xticklabels(alphab)

		plt.bar(pos, f, width, color='g')
	
		plt.show()
	except IOError:
		print "Problems opening file " + filename	



			
#p = processImages(path)
#print time.clock()
#m = fileReader.sli_from_folder_to_array(path + "/reco") 
#start = time.clock()
#f = frequencies_float(m)
#print "Time it took for freq: " + str(time.clock() - start)
#for fr in f:
#	print fr 

file_freq_float = open("frequencies_float1.txt", "w")

i = 0
for line in file_freq_float:
	file_freq_float.write(str(i) + " " + str(line) + "\n")
	i += 1
	
#histogram("frequencies_float1.txt")

plt.plotfile("frequencies_float1.txt")
plt.show()
