import time
#t = time.time()
i = 0

import xception
import numpy as np

# train.py
# Simple env test.
import json
import select
import time
import logging
import os

import aicrowd_helper
import gym
import minerl
from utility.parser import Parser

from a3c.a3c import A3C

import coloredlogs
#coloredlogs.install(logging.DEBUG)

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamond-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000))#000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 1))#5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 15))#4*24*60))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on. Make your you keep a tap on the numbers to avoid breaching any limits.
parser = Parser('performance/',
                allowed_environment=MINERL_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=True, # False
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT*60,
                initial_poll_timeout=600)

#print("time0 = ", time.time()-t)

def main():
    """
    This function will be called for training phase.
    """
    # How to sample minerl data is document here:
    # http://minerl.io/docs/tutorials/data_sampling.html
    data = minerl.data.make(MINERL_GYM_ENV, data_dir=MINERL_DATA_ROOT)

    # Sample code for illustration, add your training code below
    #env = gym.make(MINERL_GYM_ENV)

    #actions = [env.action_space.sample() for _ in range(10)] # Just doing 10 samples in this example
    xposes = []
    #print("actions = ", actions)

    #model = xception.fancy_nn()

    a3c = A3C()

    netr = 0
    for state, action, reward, next_state, done in data.sarsd_iter(num_epochs=1, max_sequence_len=32):
        #print("state =", state, ", action =", action, ", reward =", reward, ", next_state =", next_state, ", done = ", done)
        #print('whocares')

        inputs = xception.state_to_inputs(state) # returns [linear_inputs, binary_inputs]
        inputs = xception.reshape_inputs(inputs)
        #inputs = np.moveaxis(inputs, -1, 0)

        labels = xception.label_to_output(action)
        labels = xception.reshape_labels(labels)
        #labels = np.moveaxis(labels, -1, 0)

        a3c.train_models(state, action, reward, done)

        #model.fit(inputs, labels)
        #model.save("trained_network.hdf5")
        #env.render()

# run.py
import os
EVALUATION_RUNNING_ON = os.getenv('EVALUATION_RUNNING_ON', None)
EVALUATION_STAGE = os.getenv('EVALUATION_STAGE', 'training')#''all')
EXITED_SIGNAL_PATH = os.getenv('EXITED_SIGNAL_PATH', 'shared/exited')

# Training Phase
if EVALUATION_STAGE in ['training']:#['all', 'training']:
    aicrowd_helper.training_start()
    #main()
    if i == 0:
        try:
            i+=1
            main()
            aicrowd_helper.training_end()
        except Exception as e:
            print("__aicrowd_helper.training_error()__")
            aicrowd_helper.training_error()
            print(e)