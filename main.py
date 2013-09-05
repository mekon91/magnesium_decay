
import helpFunctions
import numpy as np
from processor import *





class processFolders:
	def __init__(self, path, range_files, threshholds):
		self.path = path
		folders = helpFunctions.get_folders(path) 
		print folders		
		#for each folder create an processFolder object 
		for f in folders:
			p = processFolder(path + "/" + f, range_files, threshholds)
	
	
if __name__=='__main__':
	
	
	if (len(sys.argv) == 2) :
		#default thresholds/ range files
		
		#THe threshholds for air - > decaying magnesium, decaying magnesium -> pure magnesium. 
		#These will be computed using Oscars code 
		path = sys.argv[1]
		threshholds = [103,195]
	
		#range of files to be read from the folders 
		range_files = [160,260]	
		
		processFolders(path,range_files,threshholds)
	
	elif (len(sys.argv) == 6) :
	
		path = sys.argv[1]
		range_files = [int(sys.argv[2]), int(sys.argv[3])]
		threshholds = [int(sys.argv[4]), int(sys.argv[5])]
	
		processFolders(str(path), range_files, threshholds)
	else: 
		print
		print "WRONG PARAMETERS PASSED!"
		print 
		print "Try something like: python main.py /space/hzg_w2_desy2011c/hzg09_rawbin2 160 260 103 195"
		print
		print "Or somethink like: python main.py /space/hzg_w2_desy2011c/hzg09_rawbin2 for default threshholds and range_files"
		print 
		sys.exit
		
	#example hardcoded
	
	#processFolders("/space/hzg_w2_desy2011c/hzg09_rawbin2", [160,161], [103,195])
	
	#example for command line:
	
	#processFolders(path, range_files, threshholds)
	

