from neuralnetwork.neural_net import NeuralNetwork
import csv
import pdb
import numpy as np
import pickle

'''
Feed-forward network with backprop for kaggle's digit recognizer competition
Uses pybrain (http://pybrain.org/)

Digit recognizer competition:
https://www.kaggle.com/c/digit-recognizer/

Performance so far:
0.93429 - portion of correctly labeled images (kaggle submission score)

'''


print "loading training data..."

filename = "training_data.pkl"
with open(filename, "r") as infile:
    training = pickle.load(infile)

training_x = training['x'].astype(int)
training_y = np.zeros((len(training['y']), 10))

for index, item in enumerate(training['y']):
	training_y[index][int(item)] = 1

print "done"


print "loading test data..."

filename = "test_data.pkl"
with open(filename, "r") as infile:
    test = pickle.load(infile)

test_x = test['x'].astype(int) 

print "done"


# train the network

net = NeuralNetwork(training_x.shape[1], training_y.shape[1], hidden_layer=50, learning_rate=0.05, momentum=0.6, epochs=0)
net.train(training_x, training_y)


# Evaluate training result

correct = 0
false = 0

for i in range(0, training_x.shape[0]):
	if(np.argmax(net.activate(training_x[i]),0) == np.argmax(training_y[i],0)):
		correct = correct + 1
	else:
		false = false + 1

print "correct labels:",correct
print "false labels:",false

score = correct / training_x.shape[0]
print "score: %d" % (score) # correctly labeled images


# Prepare submission for kaggle

submission = []

for i in range(0, test_x.shape[0]):
	submission.append(np.argmax(net.activate(test_x[i]),0))

headers = ['ImageId', 'Label']

with open('submission.csv', 'w') as csvfile:
    submission_writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    submission_writer.writerow(headers)
    for imageId, label in enumerate(submission):
        submission_writer.writerow([imageId+1, submission[imageId]])