import numpy as np
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
from .agent import Agent

class Actor(Agent):
    """ Actor for the A2C Algorithm
    """

    # vanilla_input = layers.Input(shape=(21,))
    # img_input = layers.Input(shape=(64, 64, 3))

    # binary_output = layers.Dense(8, name='binary_prediction')(x)
    # linear_output = layers.Dense(7, activation='linear', name='linear_prediction')(x)

    def __init__(self, network, lr):
        Agent.__init__(self, lr)
        self.model = self.addHead(network)
        self.vanilla_pl_input = K.placeholder(shape=(None, 21,))
        self.img_pl_input = K.placeholder(shape=(None, 64, 64, 3))
        self.action_pl_bin = K.placeholder(shape=(None, 8))
        self.action_pl_lin = K.placeholder(shape=(None, 8))
        self.action_pl = K.concatenate([self.action_pl_bin, self.action_pl_lin])
        self.advantages_pl = K.placeholder(shape=(None,))
        # Pre-compile for threading
        self.model._make_predict_function()

    def addHead(self, network):
        """ Assemble Actor network to predict probability of each action
        """
        x = network.output

        binary_output = Dense(8, name='binary_prediction')(x)
        linear_output = Dense(8, activation='linear', name='linear_prediction')(x)                      # needs 7

        self.linear_model = Model(inputs=network.input, outputs=linear_output)
        self.binary_model = Model(inputs=network.input, outputs=binary_output)

        model = Model(inputs=network.input, outputs=[self.binary_model.output, self.linear_model.output])

        return model

        #out = Dense(self.out_dim, activation='softmax')(x)
        #return Model(network.input, out)

    def optimizer(self):
        """ Actor Optimization: Advantages + Entropy term to encourage exploration
        (Cf. https://arxiv.org/abs/1602.01783)
        """
        weighted_actions_bin = K.sum(self.action_pl_bin * self.binary_model.output, axis=1)
        weighted_actions_lin = K.sum(self.action_pl_lin * self.linear_model.output, axis=1)
        eligibility = K.log(weighted_actions_bin + 1e-10) * K.log(weighted_actions_lin + 1e-15) * K.stop_gradient(self.advantages_pl)
        entropy = K.sum(self.linear_model.output * K.log(self.binary_model.output + 1e-10), axis=1)
        loss = 0.001 * entropy - K.sum(eligibility)

        updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], loss)
        return K.function([self.model.input, self.action_pl, self.advantages_pl], [], updates=updates)

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
