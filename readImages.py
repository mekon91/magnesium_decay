
import os,sys, scipy, warnings, time, helpFcts
import numpy as np
from os import listdir
from os.path import isfile, join
import scipy.ndimage as im
from PIL import Image, ImageSequence
from images2gif import writeGif
from regionGrowing import *

#file_extension = '.tif'
path = "/afs/desy.de/user/d/dariep/Desktop/magnesium_decay/Images"

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
		
		#creating new paths: 1 for filtered images, 1 for recolored images, 1 for gif videos, 
		#plus 4 for slices along ith and jth dimensions (2 filtered, 2 recolored)
		path_filtered = path + "/Filtered_Images"
		if not os.path.exists(path_filtered): 
			os.makedirs(path_filtered)
			
		path_recolored = path + "/Recolored_Images"
		if not os.path.exists(path_recolored):
			os.makedirs(path_recolored)
		
		#slices along the j'th (1) dimension
		path_slice_j = path + "/Slice_j_Images"
		if not os.path.exists(path_slice_j):
			os.makedirs(path_slice_j)
			
		#slices along the k'th (2) dimension
		path_slice_k = path + "/Slice_k_Images"
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
			
		#m is the 3D array containing all the images at path
		m = read_from_file_to_array(path_filtered)
		self.m = m
		
		print "Number of images: " + str(m.shape[0])		
			
		#m = apply_filters(m, path_filtered)
		
		#region grow to find the different regions.Should return list of thresholds.TODO
		threshholds = [50,185]
		
		self.m_regions = region_array(m,threshholds)
		
				
		#need to make a BASIC recolor function, one color per region, no interpolation or anything cool
		#recolor(path_filtered, path_recolored)
						
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
		helpFcts.sort_nicely(image_files)	
		return image_files
	else:
		print "No files in folder " + path
		return None
	
#returns all the files at path
def get_image_files(path):
	try:
		image_files = [ f for f in listdir(path) if isfile(join(path,f))]
	except IOError:
		"No such file or directory: "  + path 	
	#check if there are files in folder and print their names out
	if (len(image_files) >= 1):
		
		print "Images in folder: "
		for i in image_files:
			print i,
		print '\n'
		helpFcts.sort_nicely(image_files)	
		return image_files
	else:
		print "No files in folder " + path
		return None
		
def get_PIL_images(path):
	image_names = get_image_files(path)
	images = []
	for img_name in image_names:
		img = Image.open(path + "/" + img_name)
		images.append(img)
	return images
	
#using the get_image_files, it then reads all images at path with the extension 'file_extension', and stacks them on a 3D array m. m[i] is the ith image
def read_from_file_to_array(path):
	image_files = get_image_files(path)
	try:
		i = 0
		#get dimension of the image. It assumes all images are the same resolution
		first_img = Image.open(path + "/" + image_files[0])
			
		m = np.zeros((len(image_files),first_img.size[0],first_img.size[1]), dtype = int)
			
		for file in image_files:
			print "Opening image at " + path + "/" + file
			
			img = Image.open(path + "/" + file)						
			#convert the image to a 2D array. flip the x axis, because for some reason the image img is mirrored
			a = np.flipud(np.asarray(img))		
			m[i] = a
			i += 1
		print
		return m		
	except IOError:
		print "Problems opening files from" + path
		return None	

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
	print "Done	in: " + str(time.clock() - start) + " seconds"
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
def recolor(path_images, path_recolored):
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
				blue4 = (0, 0, 139)
				lawngreen = (124, 252, 0)
				black = (0,0,0)
				red2 = (238,0,0)
				mint = (189, 252, 201)

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

m = None			
p = processImages(path)
print p.m_regions

#img = Image.open('ani_img.tif')
#regiongrow(img,epsilon = 20 ,start_point = (440,50), out_img_name = 'roflstomp2.tif')

