import numpy as np
import tensorflow as tf

class ReplayMemory:

	def insert(self, state, action, reward, next_state, done):
		raise NotImplementedError

	def sample(self, batch_size):
		raise NotImplementedError

class ReplayQueue(ReplayMemory):

	def __init__(self, max_size):
		self.buffer = []
		self.max_size = max_size
		self.pointer = 0
	
	def insert(self, state, action, reward, next_state, done):
		state = np.asarray(state).reshape([-1])
		action = np.asarray(action).reshape([-1])
		reward = np.asarray(reward).reshape([-1])
		next_state = np.asarray(next_state).reshape([-1])
		
		if len(self.buffer) < self.max_size:
			self.buffer.append((state, action, reward, next_state, done))

		else:
			self.buffer[self.pointer] = (state, action, reward, next_state, done)
		
		self.pointer = (self.pointer + 1) % self.max_size
			

	def sample(self, batch_size):
		sample_indices = np.random.randint(0, len(self.buffer), size = batch_size)
		state = np.asarray([self.buffer[i][0] for i in sample_indices], dtype = np.float32).reshape([batch_size, -1])
		action = np.asarray([self.buffer[i][1] for i in sample_indices], dtype = np.float32).reshape([batch_size, -1])
		reward = np.asarray([self.buffer[i][2] for i in sample_indices], dtype = np.float32).reshape([batch_size, -1])
		next_state = np.asarray([self.buffer[i][3] for i in sample_indices], dtype = np.float32).reshape([batch_size, -1])
		done = np.asarray([self.buffer[i][4] for i in sample_indices], dtype = np.float32).reshape([batch_size, -1])

		state = tf.constant(state)
		action = tf.constant(action)
		reward = tf.constant(reward)
		next_state = tf.constant(next_state)
		done = tf.constant(done)


		return (state, action, reward, next_state, done)

