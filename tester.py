rewards = []
import time
t = time.time()
import random
i = 0

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
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))#5))
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

print("time = ", time.time()-t)

def main():
    """
    This function will be called for training phase.
    """
    # How to sample minerl data is document here:
    # http://minerl.io/docs/tutorials/data_sampling.html
    data = minerl.data.make(MINERL_GYM_ENV, data_dir=MINERL_DATA_ROOT)

    # Sample code for illustration, add your training code below
    env = gym.make(MINERL_GYM_ENV)

    actions = [env.action_space.sample() for _ in range(10)] # Just doing 10 samples in this example
    xposes = []
    print(actions)



    netr = 0
    for obs, rew, done, act in data.sarsd_iter(num_epochs=1):
        random_act = env.action_space.noop()
        random_act['camera'] = [random.uniform(-1, 2), random.uniform(-1, 2)]
        random_act['back'] = random.choice([0, 1])
        random_act['forward'] = 1
        random_act['right'] = random.choice([0, 1])
        random_act['left'] = 0  # random.choice([0,1])
        random_act['jump'] = random.choice([0, 0, 0, 0, 0, 0, 0, 1])
        random_act['attack'] = 1
        random_act['craft'] = random.choice([0, 1])
        random_act['equip'] = random.choice([0, 1])
        random_act['nearbyCraft'] = 0
        random_act['nearbySmelt'] = 0
        random_act['place'] = random.choice([0, 1])
        random_act['sneak'] = 0
        random_act['sprint'] = random.choice([0, 1])

        obs, reward, done, info = env.step(random_act)
        print("obs = ", obs)
        rewards.append(reward)

        # print(reward)
        netr += rew
        print(netr)
        env.render()

    # SKIP
    for _ in range(0):
        obs = env.reset()
        done = False
        netr = 0

        # Limiting our code to 1024 steps in this example, you can do "while not done" to run till end
        i = 0
        while (i in range(10240) and not done):
            print("step = ", i)
            i+=1

            random_act = env.action_space.noop()
            random_act['camera'] = [random.uniform(-1, 2), random.uniform(-1, 2)]
            random_act['back'] = random.choice([0,1])
            random_act['forward'] = 1
            random_act['right'] = random.choice([0,1])
            random_act['left'] = 0#random.choice([0,1])
            random_act['jump'] = random.choice([0,0,0,0,0,0,0,1])
            random_act['attack'] = 1
            random_act['craft'] = random.choice([0,1])
            random_act['equip'] = random.choice([0,1])
            random_act['nearbyCraft'] = 0
            random_act['nearbySmelt'] = 0
            random_act['place'] = random.choice([0,1])
            random_act['sneak'] = 0
            random_act['sprint'] = random.choice([0,1])

            #print(random_act)
            obs, reward, done, info = env.step(random_act)#random.choice(random_act))
            print("obs = ", obs)
            rewards.append(reward)

            #print(reward)
            netr += reward
            print(netr)
            env.render()

            # To get better view in your training phase, it is suggested
            # to register progress continuously, example when 54% completed
            # aicrowd_helper.register_progress(0.54)

            # To fetch latest information from instance manager, you can run below when you want to know the state
            #>> parser.update_information()
            #>> print(parser.payload)
            # .payload: provide AIcrowd generated json
            # Example: {'state': 'RUNNING', 'score': {'score': 0.0, 'score_secondary': 0.0}, 'instances': {'1': {'totalNumberSteps': 2001, 'totalNumberEpisodes': 0, 'currentEnvironment': 'MineRLObtainDiamond-v0', 'state': 'IN_PROGRESS', 'episodes': [{'numTicks': 2001, 'environment': 'MineRLObtainDiamond-v0', 'rewards': 0.0, 'state': 'IN_PROGRESS'}], 'score': {'score': 0.0, 'score_secondary': 0.0}}}}
            # .current_state: provide indepth state information avaiable as dictionary (key: instance id)

        print("done =", done)

    # Save trained model to train/ directory
    # Training 100% Completed
    aicrowd_helper.register_progress(1)
    env.close()

# run.py
import os
EVALUATION_RUNNING_ON = os.getenv('EVALUATION_RUNNING_ON', None)
EVALUATION_STAGE = os.getenv('EVALUATION_STAGE', 'all')
EXITED_SIGNAL_PATH = os.getenv('EXITED_SIGNAL_PATH', 'shared/exited')

# Training Phase
if EVALUATION_STAGE in ['all', 'training']:
    aicrowd_helper.training_start()
    #main()
    if i == 0:
        try:
            i+=1
            main()
            aicrowd_helper.training_end()
        except Exception as e:
            aicrowd_helper.training_error()
            print(e)