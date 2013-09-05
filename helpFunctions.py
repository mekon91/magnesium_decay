import re
from os import listdir
from os.path import isfile, isdir, join

#returns all the files at path that end with the extension file_extension
def get_files(path,range_files, file_extension):
	try:
		files = [ f for f in listdir(path) if isfile(join(path,f)) and f.endswith(file_extension)]
		#check if there are files in folder and print their names out
		
		if (len(files) >= 1):
			
			
			print "Files in the range " + str(range) + " at " + path + " with extension " + file_extension + " :" 
			i = 0;
			sort_nicely(files)	
			files_of_interest = []
			for f in files:
				if (i >= range_files[0] and i < range_files[1]):
					files_of_interest.append(f)
					print f,
				i += 1	
			print '\n'
			
			
			return files_of_interest
		else:
			print "No files in folder " + path
			return None
	except IOError:
		"No such file or directory: "  + path 	
		return None

def get_folders(path):
	try: 
		folders = [f for f in listdir(path) if not isfile(join(path,f))]
		#check if there are files in folder and print their names out
		if (len(folders) >= 1):
			print "Folders at " + path + " :" 
			for i in folders:
				print i,
			print '\n'
			sort_nicely(folders)	
			return folders
		else:
			print "No folders in at " + path
			return None		
	except IOError:
		"No such file or directory: " + path
	
	

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
