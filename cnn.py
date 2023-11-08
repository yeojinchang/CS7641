from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.layers import LeakyReLU


class CNN(object):
    def __init__(self):
        # change these to appropriate values
        self.batch_size = 16 #number of images passed to model in each pass - a power of 2
        self.epochs = 2 #number of epochs to train over
        self.init_lr= 1e-2 #learning rate - this will be a small number

        # No need to modify these
        self.model = None

    def get_vars(self):
        return self.batch_size, self.epochs, self.init_lr

    def create_net(self):
        '''
        In this function you are going to build a convolutional neural network based on TF Keras.
        First, use Sequential() to set the inference features on this model. 
        Then, use model.add() to build layers in your own model
        Return: model
        '''

        #TODO: implement this

        
        return self.model

    def compile_net(self, model):
        '''
        In this function you are going to compile the model you've created.
        Use self.model.compile() to build your model.
        '''
        self.model = model

        #TODO: implement this
        

        return self.model