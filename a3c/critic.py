import numpy as np
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
from .agent import Agent

class Critic(Agent):
    """ Critic for the A3C Algorithm
    """

    def __init__(self, network, lr):
        Agent.__init__(self, lr)
        self.model = self.addHead(network)
        self.discounted_r_bin = K.placeholder(shape=(None,))
        self.discounted_r_lin = K.placeholder(shape=(None,))
        self.discounted_r = K.concatenate([self.discounted_r_bin, self.discounted_r_lin])
        # Pre-compile for threading
        self.model._make_predict_function()

    def addHead(self, network):
        """ Assemble Critic network to predict value of each state
        """
        x = network.output

        binary_output = Dense(8, name='binary_prediction')(x)
        linear_output = Dense(8, activation='linear', name='linear_prediction')(x)

        self.linear_model = Model(inputs=network.input, outputs=linear_output)
        self.binary_model = Model(inputs=network.input, outputs=binary_output)

        model = Model(inputs=network.input, outputs=[self.binary_model.output, self.linear_model.output])
        return model

    def optimizer(self):
        """ Critic Optimization: Mean Squared Error over discounted rewards
        """
        critic_loss = K.mean(K.square(self.discounted_r - self.model.output))
        updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], critic_loss)
        return K.function([self.model.input, self.discounted_r], [], updates=updates)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
