import gym
import numpy as np
from ddpg import DDPG
from replay_buffer import ReplayQueue

import tensorflow as tf
from tensorflow import keras

env = gym.make('Pendulum-v0')

def get_pi(state_dims, hidden_units, action_dims):
    model = keras.Sequential([
        keras.layers.Dense(hidden_units, activation = tf.nn.relu),
        keras.layers.Dense(hidden_units, activation = tf.nn.relu),
        keras.layers.Dense(action_dims, activation = tf.nn.tanh),
        keras.layers.Lambda(lambda x: env.action_space.high * x)
    ])
    random_input = np.random.normal(size = [1, env.observation_space.shape[0]])
    model(random_input)
    return model

def get_q(state_dims, hidden_units, action_dims):
    model = keras.Sequential([
        keras.layers.Dense(hidden_units, activation = tf.nn.relu),
        keras.layers.Dense(hidden_units, activation = tf.nn.relu),
        keras.layers.Dense(1)
    ])
    random_input = np.random.normal(size = [1, env.action_space.shape[0] + env.observation_space.shape[0]])
    model(random_input)
    return model




pi = get_pi(state_dims = env.observation_space.shape[0], hidden_units = 256, action_dims = env.action_space.shape[0])
tpi = get_pi(state_dims = env.observation_space.shape[0], hidden_units = 256, action_dims = env.action_space.shape[0])

q = get_q(state_dims = env.observation_space.shape[0], hidden_units = 256, action_dims = env.action_space.shape[0])
tq = get_q(state_dims = env.observation_space.shape[0], hidden_units = 256, action_dims = env.action_space.shape[0])


GAMMA = 0.99
RHO = 0.995
SIGMA = 0.1
BATCH_SIZE = 64

pi_optimizer = keras.optimizers.Adam(0.001)
q_optimizer = keras.optimizers.Adam(0.002)


MAX_BUFFER_SIZE = 100000
INIT_BUFFER_SIZE = 4098

replay_memory = ReplayQueue(MAX_BUFFER_SIZE)

class Trig:
	def __init__(self):
		self.cnt = 0
		self.pi = pi
		self.test_env = gym.make('Pendulum-v0')
	
	def __call__(self):
		self.cnt += 1
		state = self.test_env.reset()
		self.test_env.render()
		done = False
		treward = 0
		while not done:
			action = pi(np.asarray(state).reshape([1, -1]))
			state, reward, done, indo = self.test_env.step(action[0].numpy())
			self.test_env.render()
			treward += reward
		
		print('Episode', self.cnt, 'Total Reward', treward)

trigger = Trig()
ddpg = DDPG(pi, q, tpi, tq, pi_optimizer, q_optimizer, env, replay_memory, GAMMA, RHO, SIGMA, 64, 4098, trigger)

while trigger.cnt < 200:

	ddpg.step()
