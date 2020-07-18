import tensorflow as tf
import numpy as np
from tensorflow import keras


def polyak_average(target_weights, weights, rho):
	result = []
	for tw, w in zip(target_weights, weights):
		result.append(rho * tw + (1. - rho) * w)
	return result



class DDPG:
	def __init__(self, pi, q, tpi, tq, pi_optimizer, q_optimizer, env, replay_buffer, gamma = 0.99, rho = 0.995, sigma = 0.1, batch_size = 64, initial_buffer_size = 100000, trigger = None):
		self.policy_net = pi
		self.value_net = q
		self.target_policy_net = tpi
		self.target_value_net = tq
		self.pi_optimizer = pi_optimizer
		self.q_optimizer = q_optimizer
		self.sigma = sigma
		self.env = env
		self.batch_size = batch_size
		self.done = False
		self.state = self.env.reset()
		self.replay_buffer = replay_buffer
		self.rho = rho
		self.gamma = gamma
		self.trigger = trigger

		tpi.set_weights(pi.get_weights())
		tq.set_weights(q.get_weights())

		for _ in range(initial_buffer_size):
			input_tensor = tf.constant(np.asarray(self.state, dtype = np.float64).reshape([1, -1]))
			action = self.policy_net(input_tensor)
			action = tf.cast(tf.reshape(action, shape = [-1]), dtype = tf.float64) + tf.random.normal(stddev = self.sigma, shape = self.env.action_space.shape, dtype = tf.float64)
			action = tf.clip_by_value(action, self.env.action_space.low, self.env.action_space.high)
			new_state, reward, done, _ = self.env.step(action)
			self.done = done
			self.replay_buffer.insert(*(self.state, action[0].numpy(), reward, new_state, done))
			self.state = new_state
			if done:
				self.state = self.env.reset()

		print('Populated replay buffer with', initial_buffer_size, 'samples')



	def sample_transition(self, noise_stddev):
		action = self.policy_net(np.asarray(self.state).reshape([1, -1]))
		action = tf.cast(tf.reshape(action, shape = [-1]), dtype = tf.float64) + tf.random.normal(stddev = noise_stddev, shape = self.env.action_space.shape, dtype = tf.float64)
		action = tf.clip_by_value(action, self.env.action_space.low, self.env.action_space.high)
		new_state, reward, done, _ = self.env.step(action)
		self.done = done
		self.replay_buffer.insert(*(self.state, action[0].numpy(), reward, new_state, done))
		self.state = new_state
		if done:
			self.state = self.env.reset()
				
	@tf.function
	def policy_objective(self, starts, actions, rewards, ends, done):
		action_head = self.policy_net(starts)
		value_head = self.value_net(tf.concat([starts, action_head], axis = -1))
		return -tf.reduce_mean(value_head)
		
	
	@tf.function
	def value_objective(self, starts, actions, rewards, ends, done):
		# target(st, a) = rt + gamma * Qpi_target(st+1, pi_target(st+1)) * (1. - done)
		next_value = self.target_value_net(tf.concat([ends, self.target_policy_net(ends)], axis = -1))
		target = rewards + self.gamma * next_value * (1. - done)
		# print(tf.reduce_sum(tf.cast(tf.math.is_nan(target), tf.int32)), tf.reduce_sum(tf.cast(tf.math.is_nan(rewardstargei), tf.int32)), )
		current_value = self.value_net(tf.concat([starts, actions], axis = -1))
		return tf.reduce_mean(tf.square(current_value - target))

	
	def soft_update_target(self):
		self.target_policy_net.set_weights(polyak_average(self.target_policy_net.get_weights(), self.policy_net.get_weights(), self.rho))
		self.target_value_net.set_weights(polyak_average(self.target_value_net.get_weights(), self.value_net.get_weights(), self.rho))

	def step(self, **kwargs):
		self.sample_transition(self.sigma)
		samples = self.replay_buffer.sample(self.batch_size)
		with tf.GradientTape() as tape:
			qloss = self.value_objective(*samples)
		
		grads = tape.gradient(qloss, self.value_net.trainable_variables)
		self.q_optimizer.apply_gradients(zip(grads, self.value_net.trainable_variables))

		with tf.GradientTape() as tape:
			piloss = self.policy_objective(*samples)


		grads = tape.gradient(piloss, self.policy_net.trainable_variables)
		self.pi_optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

		if self.done and self.trigger is not None:
			self.trigger(**kwargs)

		self.soft_update_target()

