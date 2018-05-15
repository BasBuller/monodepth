import random
import sys

def random_sampler(filename, k):
	"""
	Function that selects k lines randomly from a file (of train images)
	"""

	sample = []

	with open(filename, 'rb') as f:

		f.seek(0, 2)
		filesize = f.tell()

		random_set = sorted(random.sample(range(filesize), k))

		for i in range(k):

			f.seek(random_set[i])

			# Skip current line (because we might be in the middle of a line) 
			f.readline()

			# Append the next line to the sample set 
			sample.append(f.readline().rstrip())

	with open(filename.split('.')[0]+'_random.txt', 'w') as f:

		for s in sample:
			f.write(s.decode('ascii')+'\n')

if __name__ == '__main__':
	
	random_sampler(sys.argv[1], int(sys.argv[2]))