import warnings
warnings.filterwarnings('ignore')

import xception
from keras import models

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

import coloredlogs
coloredlogs.install(logging.DEBUG)

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
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT*60,
                initial_poll_timeout=600)

def first_step(env):
    import random
    random_act = env.action_space.noop()
    random_act['camera'] = [0.1, 1]  # [random.uniform(-1, 2), random.uniform(-1, 2)]
    random_act['back'] = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # [0,1]
    random_act['forward'] = 1
    random_act['right'] = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # [0,1]
    random_act['left'] = 0  # random.choice([0,1])
    random_act['jump'] = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    random_act['attack'] = 1
    random_act['craft'] = random.choice([0, 1])
    random_act['equip'] = random.choice([0, 1])
    random_act['nearbyCraft'] = 0
    random_act['nearbySmelt'] = 0
    random_act['place'] = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # [0,1]
    random_act['sneak'] = 0
    random_act['sprint'] = random.choice([0, 1])



def main():
    i=0
    netr = 0

    env = gym.make(MINERL_GYM_ENV)

    model = models.load_model('pretrained_network.hdf5')

    obs, reward, done, info = env.step(first_step(env))

    while (i in range(10240) and not done):
        inputs = xception.state_to_inputs(obs)  # returns [linear_inputs, binary_inputs]
        inputs = xception.reshape_inputs(inputs)

        obs, reward, done, info = env.step(
            xception.outputs_to_action_7(
                model.predict(inputs)))

        netr += reward
        print(netr)
        env.render()

    env.close()

i=0
if i == 0:
    try:
        i += 1
        aicrowd_helper.training_start()
        main()
        aicrowd_helper.training_end()
    except Exception as e:
        if str(e).__contains__('An attempt has been made to start a new process before the'):
            pass
        else:
            raise e