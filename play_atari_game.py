#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random as rand
from collections import deque


class Memory:

    def __init__(self, size, batch_size):
        self.batch_size = batch_size
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)
        
    def add(self, state, action, reward, next_state, terminal):
        if len(self.memory) >= self.memory.maxlen:
            self.memory.popleft()
        self.memory.append( (state, action, reward, next_state, terminal) )

    def getSample(self):
        return rand.sample(self.memory, self.batch_size)

    def reset(self):
        self.memory.clear()


# In[2]:


import tensorflow as tf
class ConvNet:
    
    def __init__(self, params, trainable):
        self.shape = [None, params.width, params.height, params.history_length]
        self.x = tf.placeholder(tf.float32, self.shape)
        self.in_dims = self.shape[1]*self.shape[2]*self.shape[3]
        self.out_dims = params.actions
        self.filters = [32, 64, 64] # convolution filters at each layer
        self.num_layers = 3 # number of convolutional layers
        self.filter_size = [8, 4, 4] # size at each layer
        self.filter_stride = [4, 2, 1] # stride at each layer
        self.fc_size = [512] # size of fully connected layers
        self.fc_layers = 1 # number of fully connected layers
        self.trainable = trainable

        # dictionary for weights in network
        self.weights = {}
        # get predicted activation
        self.y = self.infer(self.x)

    def create_weight(self, shape):
        init = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(init, name='weight')

    def create_bias(self, shape):
        init = tf.constant(0.01, shape=shape)
        return tf.Variable(init, name='bias')

    def create_conv2d(self, x, w, stride):
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

    def max_pool(self, x, size):
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

    def infer(self, _input):
        self.layers = [_input]

        # initialize convolution layers
        for layer in range(self.num_layers):
            with tf.variable_scope('conv' + str(layer)) as scope:
                if layer == 0:
                    in_channels = self.shape[-1]
                    out_channels = self.filters[layer]
                else:
                    in_channels = self.filters[layer-1]
                    out_channels = self.filters[layer]

                shape = [ self.filter_size[layer], 
                          self.filter_size[layer],
                          in_channels, 
                          out_channels ]

                w = self.create_weight(shape)
                conv = self.create_conv2d(self.layers[-1], w, self.filter_stride[layer])

                b = self.create_bias([out_channels])
                self.weights[w.name] = w
                self.weights[b.name] = b
                bias = tf.nn.bias_add(conv, b)
                conv = tf.nn.relu(bias, name=scope.name)
                self.layers.append(conv)

        last_conv = self.layers[-1]

        # flatten last convolution layer
        dim = 1
        for d in last_conv.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(last_conv, [-1, dim], name='flat')
        self.layers.append(reshape)

        # initialize fully-connected layers
        for layer in range(self.fc_layers):
            with tf.variable_scope('hidden' + str(layer)) as scope:
                if layer == 0:
                    in_size = dim
                else:
                    in_size = self.fc_size[layer-1]

                out_size = self.fc_size[layer]
                shape = [in_size, out_size]
                w = self.create_weight(shape)
                b = self.create_bias([out_size])
                self.weights[w.name] = w
                self.weights[b.name] = b
                hidden = tf.nn.relu_layer(self.layers[-1], w, b, name=scope.name)
                self.layers.append(hidden)

        # create last fully-connected layer
        with tf.variable_scope('output') as scope:
            in_size = self.fc_size[self.fc_layers - 1]
            out_size = self.out_dims
            shape = [in_size, out_size]
            w = self.create_weight(shape)
            b = self.create_bias([out_size])
            self.weights[w.name] = w
            self.weights[b.name] = b
            hidden = tf.nn.bias_add(tf.matmul(self.layers[-1], w), b)
            self.layers.append(hidden)

        # return activation of the network
        return self.layers[-1]


# In[3]:


import numpy as np
class Buffer:

    def __init__(self, params):
        history_length = params.history_length
        width = params.width
        height = params.height
        self.dims = (width, height, history_length)
        self.buffer = np.zeros(self.dims, dtype=np.uint8)

    def add(self, state):
        self.buffer[:, :, :-1] = self.buffer[:, :, 1:]
        self.buffer[:, :, -1] = state

    def getInput(self):
        x = np.reshape(self.buffer, (1,)+ self.dims)
        return x

    def getState(self):
        return self.buffer

    def reset(self):
        self.buffer.fill(0)


# In[4]:


import time
import tensorflow as tf


class Trainer:

    def __init__(self, agent):
        self.agent = agent
        self.env = agent.env
        self.saver = tf.train.Saver()

    def run(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.agent.randomRestart()

            successes = 0
            failures = 0
            total_loss = 0

            print("starting %d random plays to populate replay memory" % self.agent.replay_start_size)
            for i in range(self.agent.replay_start_size):
                # follow random policy
                state, action, reward, next_state, terminal = self.agent.observe(1)

                if reward == 1:
                    successes += 1
                elif terminal:
                    failures += 1

                if (i+1) % 10000 == 0:
                    print ("\nmemory size: %d" % len(self.agent.memory),                          "\nSuccesses: ", successes,                          "\nFailures: ", failures)
            
            sample_success = 0
            sample_failure = 0
            print("\nstart training...")
            start_time = time.time()
            for i in range(self.agent.train_steps):
                # annealing learning rate
                lr = self.agent.trainEps(i)
                state, action, reward, next_state, terminal = self.agent.observe(lr)

                if len(self.agent.memory) > self.agent.batch_size and (i+1) % self.agent.update_freq == 0:
                    sample_success, sample_failure, loss = self.agent.doMinibatch(sess, sample_success, sample_failure)
                    total_loss += loss

                if (i+1) % self.agent.steps == 0:
                    self.agent.copy_weights(sess)

                if reward == 1:
                    successes += 1
                elif terminal:
                    failures += 1
                
                if ((i+1) % self.agent.save_weights == 0):
                    self.agent.save(self.saver, sess, i+1)

                if ((i+1) % self.agent.batch_size == 0):
                    avg_loss = total_loss / self.agent.batch_size
                    end_time = time.time()
                    print ("\nTraining step: ", i+1,                          "\nmemory size: ", len(self.agent.memory),                          "\nLearning rate: ", lr,                          "\nSuccesses: ", successes,                          "\nFailures: ", failures,                          "\nSample successes: ", sample_success,                          "\nSample failures: ", sample_failure,                          "\nAverage batch loss: ", avg_loss,                          "\nBatch training time: ", (end_time-start_time)/self.agent.batch_size, "s")
                    start_time = time.time()
                    total_loss = 0


# In[5]:


import random as rand
import tensorflow as tf
import numpy as np
class DQN:

    def __init__(self, env, params):
        self.env = env
        params.actions = env.actions()
        self.num_actions = env.actions()
        self.episodes = params.episodes
        self.steps = params.steps
        self.train_steps = params.train_steps
        self.update_freq = params.update_freq
        self.save_weights = params.save_weights
        self.history_length = params.history_length
        self.discount = params.discount
        self.eps = params.init_eps
        self.eps_delta = (params.init_eps - params.final_eps) / params.final_eps_frame
        self.replay_start_size = params.replay_start_size
        self.eps_endt = params.final_eps_frame
        self.random_starts = params.random_starts
        self.batch_size = params.batch_size
        self.ckpt_file = params.ckpt_dir+'/'+params.game

        self.global_step = tf.Variable(0, trainable=False)
        if params.lr_anneal:
            self.lr = tf.train.exponential_decay(params.lr, self.global_step, params.lr_anneal, 0.96, staircase=True)
        else:
            self.lr = params.lr

        self.buffer = Buffer(params)
        self.memory = Memory(params.size, self.batch_size)

        with tf.variable_scope("train") as self.train_scope:
            self.train_net = ConvNet(params, trainable=True)
        with tf.variable_scope("target") as self.target_scope:
            self.target_net = ConvNet(params, trainable=False)

        self.optimizer = tf.train.RMSPropOptimizer(self.lr, params.decay_rate, 0.0, self.eps)

        self.actions = tf.placeholder(tf.float32, [None, self.num_actions])
        self.q_target = tf.placeholder(tf.float32, [None])
        self.q_train = tf.reduce_max(tf.multiply(self.train_net.y, self.actions), reduction_indices=1)
        self.diff = tf.subtract(self.q_target, self.q_train)

        half = tf.constant(0.5)
        if params.clip_delta > 0:
            abs_diff = tf.abs(self.diff)
            clipped_diff = tf.clip_by_value(abs_diff, 0, 1)
            linear_part = abs_diff - clipped_diff
            quadratic_part = tf.square(clipped_diff)
            self.diff_square = tf.multiply(half, tf.add(quadratic_part, linear_part))
        else:
            self.diff_square = tf.multiply(half, tf.square(self.diff))

        if params.accumulator == 'sum':
            self.loss = tf.reduce_sum(self.diff_square)
        else:
            self.loss = tf.reduce_mean(self.diff_square)

        # backprop with RMS loss
        self.task = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def randomRestart(self):
        self.env.restart()
        for _ in range(self.random_starts):
            action = rand.randrange(self.num_actions)
            reward = self.env.act(action)
            state = self.env.getScreen()
            terminal = self.env.isTerminal()
            self.buffer.add(state)

            if terminal:
                self.env.restart()

    def trainEps(self, train_step):
        if train_step < self.eps_endt:
            return self.eps - train_step * self.eps_delta
        else:
            return self.eps_endt

    def observe(self, exploration_rate):
        if rand.random() < exploration_rate:
            a = rand.randrange(self.num_actions)
        else:
            x = self.buffer.getInput()
            action_values = self.train_net.y.eval( feed_dict={ self.train_net.x: x } )
            a = np.argmax(action_values)
        
        state = self.buffer.getState()
        action = np.zeros(self.num_actions)
        action[a] = 1.0
        reward = self.env.act(a)
        screen = self.env.getScreen()
        self.buffer.add(screen)
        next_state = self.buffer.getState()
        terminal = self.env.isTerminal()

        reward = np.clip(reward, -1.0, 1.0)

        self.memory.add(state, action, reward, next_state, terminal)
        
        
        return state, action, reward, next_state, terminal

    def doMinibatch(self, sess, successes, failures):
        batch = self.memory.getSample()
        state = np.array([batch[i][0] for i in range(self.batch_size)]).astype(np.float32)
        actions = np.array([batch[i][1] for i in range(self.batch_size)]).astype(np.float32)
        rewards = np.array([batch[i][2] for i in range(self.batch_size)]).astype(np.float32)
        successes += np.sum(rewards==1)
        next_state = np.array([batch[i][3] for i in range(self.batch_size)]).astype(np.float32)
        terminals = np.array([batch[i][4] for i in range(self.batch_size)]).astype(np.float32)

        failures += np.sum(terminals==1)
        q_target = self.target_net.y.eval( feed_dict={ self.target_net.x: next_state } )
        q_target_max = np.argmax(q_target, axis=1)
        q_target = rewards + ((1.0 - terminals) * (self.discount * q_target_max))

        (result, loss) = sess.run( [self.task, self.loss],
                                    feed_dict={ self.q_target: q_target,
                                                self.train_net.x: state,
                                                self.actions: actions } )

        return successes, failures, loss

    def play(self):
        self.randomRestart()
        self.env.restart()
        for i in range(self.episodes):
            terminal = False
            while not terminal:
                #aca cambie algo
                state, action, reward, screen, terminal = self.observe(self.eps)

    def copy_weights(self, sess):
        for key in self.train_net.weights.keys():
            t_key = 'target/' + key.split('/', 1)[1]
            sess.run(self.target_net.weights[t_key].assign(self.train_net.weights[key]))

    def save(self, saver, sess, step):
        saver.save(sess, self.ckpt_file, global_step=step)
        
    def restore(self, saver):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_file)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)


# In[6]:


import gym
import cv2

class Environment:

    def __init__(self, params):
        self.gym = gym.make(params.game)
        self.observation = None
        self.display = params.display
        self.terminal = False
        self.dims = (params.height, params.width)

    def actions(self):
        return self.gym.action_space.n

    def restart(self):
        self.observation = self.gym.reset()
        self.terminal = False

    def act(self, action):
        if self.display:
            self.gym.render()
        self.observation, reward, self.terminal, info = self.gym.step(action)
        if self.terminal:
            #if self.display:
            #    print "No more lives, restarting"
            self.gym.reset()
        return reward

    def getScreen(self):
        return cv2.resize(cv2.cvtColor(self.observation, cv2.COLOR_RGB2GRAY), self.dims)

    def isTerminal(self):
        return self.terminal


# In[7]:


import os
import argparse
import random as rand

## these are our command line arguments.
parser = argparse.ArgumentParser()
envarg = parser.add_argument_group('Environment')
envarg.add_argument("--game", type=str, default="SpaceInvaders-v0", help="Name of the atari game to test")
envarg.add_argument("--width", type=int, default=84, help="Screen width")
envarg.add_argument("--height", type=int, default=84, help="Screen height")

memarg = parser.add_argument_group('Memory')
memarg.add_argument("--size", type=int, default=100000, help="Memory size.")
memarg.add_argument("--history_length", type=int, default=4, help="Number of most recent frames experiences by the agent.")

dqnarg = parser.add_argument_group('DQN')
dqnarg.add_argument("--lr", type=float, default=0.00025, help="Learning rate.")
dqnarg.add_argument("--lr_anneal", type=float, default=20000, help="Step size of learning rate annealing.")
dqnarg.add_argument("--discount", type=float, default=0.99, help="Discount rate.")
dqnarg.add_argument("--batch_size", type=int, default=32, help="Batch size.")
dqnarg.add_argument("--accumulator", type=str, default='mean', help="Batch accumulator.")
dqnarg.add_argument("--decay_rate", type=float, default=0.95, help="Decay rate for RMSProp.")
dqnarg.add_argument("--min_decay_rate", type=float, default=0.01, help="Min decay rate for RMSProp.")
dqnarg.add_argument("--init_eps", type=float, default=1.0, help="Initial value of e in e-greedy exploration.")
dqnarg.add_argument("--final_eps", type=float, default=0.1, help="Final value of e in e-greedy exploration.")
dqnarg.add_argument("--final_eps_frame", type=float, default=1000000, help="The number of frames over which the initial value of e is linearly annealed to its final.")
dqnarg.add_argument("--clip_delta", type=float, default=1, help="Clip error term in update between this number and its negative.")
dqnarg.add_argument("--steps", type=int, default=10000, help="Copy main network to target network after this many steps.")
dqnarg.add_argument("--train_steps", type=int, default=500000, help="Number of training steps.")
dqnarg.add_argument("--update_freq", type=int, default=4, help="The number of actions selected between successive SGD updates.")
dqnarg.add_argument("--replay_start_size", type=int, default=50000, help="A uniform random policy is run for this number of frames before learning starts and the resulting experience is used to populate the replay memory.")
dqnarg.add_argument("--save_weights", type=int, default=10000, help="Save the mondel after this many steps.")

testarg = parser.add_argument_group('Test')
testarg.add_argument("--display", dest="display", help="Display screen during testing.")
testarg.set_defaults(display=False)
testarg.add_argument("--random_starts", type=int, default=30, help="Perform max this number of no-op actions to be performed by the agent at the start of an episode.")
testarg.add_argument("--ckpt_dir", type=str, default='model', help="Tensorflow checkpoint directory.")
testarg.add_argument("--out", help="Output directory for gym.")
testarg.add_argument("--episodes", type=int, default=100, help="Number of episodes.")
testarg.add_argument("--seed", type=int, help="Random seed.")
args = parser.parse_args()
if args.seed:
    rand.seed(args.seed)
if not os.path.exists(args.ckpt_dir):
	os.makedirs(args.ckpt_dir)

#Checking for/Creating gym output directory
if args.out:
	if not os.path.exists(args.out):
		os.makedirs(args.out)
else:
	if not os.path.exists('gym-out/' + args.game):
		os.makedirs('gym-out/' + args.game)
	args.out = 'gym-out/' + args.game

##Now let's train...

#example of reinforcement learning in a game environment#
#Q learning is a gernalised AI tech which builds a model of the environment 
#without prior knowledge via the use of a experenced relay
#experienced relay is a automated learning technique where past experence 
#are incorporated into future models

#initialise gym environment and dqn
env = Environment(args)
agent = DQN(env, args)

# train agent
Trainer(agent).run()

# play the game
env.gym.monitor.start(args.out, force=True)
agent.play()
env.gym.monitor.close()
#run: python play_atari_game.py --display true
#--display true allows you to view the game being played


# In[ ]:




