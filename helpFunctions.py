import re
from os import listdir
from os.path import isfile, join

#returns all the files at path that end with the extension file_extension
def get_files(path,file_extension):
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
		sort_nicely(files)	
		return files
	else:
		print "No files in folder " + path
		return None

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
