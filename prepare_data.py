import csv
import pdb
import numpy as np
import pickle

'''
Reads kaggle's train and test data in and saves it pickled as np.arrays

data can be found at: https://www.kaggle.com/c/digit-recognizer/data
download and put into folder "data"
'''

print "reading training data..."

training_x = np.array([])
training_y = []

with open('data/train.csv', 'rb') as csvfile:
	reader =  csv.reader(csvfile, delimiter=',')
	reader.next() # skip the column labels
	first_row = reader.next()
	training_y.append(first_row.pop(0)) # first label
	training_x = np.array(first_row) # first image

	for row in reader:
		training_y.append(row.pop(0)) 
		training_x = np.vstack((training_x, row))

training_dump = {'x': training_x, 'y': training_y} # x is np.array, y is a list

filename = "training_data.pkl"
with open(filename,"w+") as outfile:
     pickle.dump(training_dump,outfile)

print "done"


print "reading test data..."

test_x = np.array([])

with open('data/test.csv', 'rb') as csvfile:
	reader =  csv.reader(csvfile, delimiter=',')
	reader.next() # skip the column labels
	first_row = reader.next()
	test_x = np.array(first_row) # first image

	for row in reader:
		test_x = np.vstack((test_x, row))

test_dump = {'x': test_x} # x is np.array

filename = "test_data.pkl"
with open(filename,"w+") as outfile:
     pickle.dump(test_dump,outfile)

print "done"