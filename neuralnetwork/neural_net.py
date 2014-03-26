import numpy as np
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, GaussianLayer, FullConnection
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import pdb
import pickle
import time

class NeuralNetwork:
    """
    Implements a full-conected feed-forward network for classification with backpropagation using pybrain
    """

    def __init__(self, inputs, targets, hidden_layer, learning_rate, momentum, epochs):
        """ Init the neural network and either loads previously saved weights or initializes new weights 
            learning_rate and momentum -- pass through to the backprop trainer
            epochs -- number of iterations we want to train. set to 0 to train until convergence 
                      (stop criterion: error does not improve for 10 epochs)
        """

        self.hidden_layer = hidden_layer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        
        try:
            with open('network.dump') as file:
                self.n = pickle.load(open('network.dump'))
                self.n.sorted = False
                self.n.sortModules()
                print "loaded existing network"

        except IOError as e:
            self.init_network(inputs, targets); 
            pickle.dump(self.n, open('network.dump', 'w'))
            print "created new network"

    def activate(self, x):
        return self.n.activate(x)

    def train(self, x, y):
        ''' Trains on the given inputs and labels for either a fixed number of epochs or until convergence.
            Normalizes the input with a z-transform'''

        print "training..."
        
        # normalize input
        m = x.mean()
        s = x.std()
        x = self.z_transform(x, m, s)

        ds = SupervisedDataSet(x.shape[1], 1) 
        ds.setField('input', x)
        ds.setField('target', y)
        
        trainer = BackpropTrainer(self.n,ds, learningrate=self.learning_rate, momentum=self.momentum, verbose=True)

        if (self.epochs == 0):
            trainer.trainUntilConvergence()
        else:
            for i in range(0, self.epochs):
                start_time = time.time()
                trainer.train() 
                print "epoch: ", i
                print "time: ", time.time() - start_time, " seconds"
            
        print "finished"
        
    def init_network(self, inputs, targets):
        """ Build the network as a full-connected feed-forward net.
        Uses the softmax function for the output layer as recommended for classification"""

        self.n = FeedForwardNetwork()

        inLayer = LinearLayer(inputs, name='in' )
        hiddenLayer = SigmoidLayer(self.hidden_layer, name='hidden')
        outLayer = SoftmaxLayer(targets, name='out')
        
        self.n.addInputModule(inLayer)
        self.n.addModule(hiddenLayer)
        self.n.addOutputModule(outLayer)
        
        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_out = FullConnection(hiddenLayer, outLayer)
        
        self.n.addConnection(in_to_hidden)
        self.n.addConnection(hidden_to_out)
        
        self.n.sortModules()


    def z_transform(self, x,m,s):
        return (x-float(m))/float(s)