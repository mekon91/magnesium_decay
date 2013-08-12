class Queue:
	"""Simple queue class
	"""
	
	def __init__(self):
		self.items = []
	 
	def isEmpty(self):
		return self.items==[]
	 
	def enque(self,item):
		self.items.insert(0,item)
	 
	def deque(self):
		return self.items.pop()
	 
	def qsize(self):
		return len(self.items)
	
	def isInside(self, item):
		return (item in self.items)
 
import Image,os
 
def regiongrow(image,epsilon,start_point,out_img_name):
 
	Q = Queue()
	s = []
	x = start_point[0]
	y = start_point[1]
	start_gray = image.getpixel( (x,y) )
	#converts to grayscale (i think). Can remove for this project
	#image = image.convert("L")
	Q.enque((x,y))
	 
	while not Q.isEmpty():
	 
		t = Q.deque()
		x = t[0]
		y = t[1]
		
		if x < image.size[0]-1 and \
		 abs( image.getpixel( (x + 1 , y) ) - start_gray ) <= epsilon :
	 
			if not Q.isInside( (x + 1 , y) ) and not (x + 1 , y) in s:
				Q.enque( (x + 1 , y) )
			
	 
		if x > 0 and \
		 abs( image.getpixel( (x - 1 , y) ) - start_gray ) <= epsilon:
	 
			if not Q.isInside( (x - 1 , y) ) and not (x - 1 , y) in s:
				Q.enque( (x - 1 , y) )
				
	 
		if y < (image.size[1] - 1) and \
		 abs( image.getpixel( (x , y + 1) ) - start_gray ) <= epsilon:
	 
			if not Q.isInside( (x, y + 1) ) and not (x , y + 1) in s:
				Q.enque( (x , y + 1) )
			
	 
		if y > 0 and \
		 abs( image.getpixel( (x , y - 1) ) - start_gray) <= epsilon:
	 
			if not Q.isInside( (x , y - 1) ) and not (x , y - 1) in s:
				Q.enque( (x , y - 1) )
			
	 
	 
		if t not in s: 
			s.append( t )
	
	out_img = Image.new("RGB", image.size, "black")
	out_img.load()
	putpixel = out_img.im.putpixel
	for i in range ( out_img.size[0] ):
		for j in range ( out_img.size[1] ):
			#out_img.im.
			putpixel( (i , j) , 255 )
 
	for i in s:
		#out_img.im.putpixel(i,10)
		putpixel(i , 150)
	out_img.save(out_img_name)	
	

