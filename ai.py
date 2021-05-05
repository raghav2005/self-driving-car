# ai for self driving car - python 2.7.13

# import all libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# create neural network architecture
class neural_network(nn.Module):

	def __init__(self, input_neuron_num, output_neuron_num):
		super(neural_network, self).__init__()

		self.input_neuron_num = input_neuron_num 
		self.output_neuron_num = output_neuron_num
		self.hidden_neuron_num = 30
		self.in_to_hid_full_connection = nn.Linear(input_neuron_num, \
			hidden_neuron_num)
		self.hid_to_out_full_connection = nn.Linear(hidden_neuron_num, \
			output_neuron_num)
		
	def forward_propagation(self, input_state):
		activated_hidden_neurons = functional.relu( \
			self.in_to_hid_full_connection(input_state))
		output_neuron_q_vals = self.hid_to_out_full_connection( \
			activated_hidden_neurons)
		
		return output_neuron_q_vals

# implement experience replay
class memory_replay(object):

	def __init__(self, capacity_of_events):
		self.capacity_of_events = capacity_of_events
		self.memory_of_events = []

	def push_to_memory(self, event):
		self.memory_of_events.append(event)

		if len(self.memory_of_events) > self.capacity_of_events:
			del self.memory_of_events[0]

	def get_sample(self, batch_size):
		samples_in_memory = zip(*random.sample(self.memory_of_events, \
			batch_size))
		
		return map(lambda to_torch_var: Variable(torch.cat( \
			to_torch_var, 0)), samples_in_memory)

# implement deep q-learning
class deep_q_network():
	
	def __init__(self, input_neuron_num, output_neuron_num, gamma):
		self.gamma = gamma
		self.reward_window = []
		self.neural_network_model = neural_network(input_neuron_num, \
			output_neuron_num)
		self.memory_of_events = memory_replay(100000)
		self.optimizer = optim.Adam(self.neural_network_model.parameters(), \
			lr = 0.001)
		self.last_state = torch.Tensor(input_neuron_num).unsqueeze(0)
		self.last_action = 0
		self.last_reward = 0
	
	def select_action(self, state):
		probabilities = functional.softmax(self.model(Variable(state, \
			volatile = True)) * 7)
		action = probabilities.multinomial()

		return action.data[0, 0]

	def learn(self, batch_state, batch_state_next, batch_reward, batch_action):
		outputs = self.neural_network_model(batch_state).gather( \
			1, batch_action.unsqueeze(1)).squeeze(1)
		next_outputs = self.neural_network_model(batch_state_next \
			).detach().max(1)[0]

		target = self.gamma * next_outputs + batch_reward
		temporal_difference_loss = functional.smooth_l1_loss(outputs, \
			target)

		self.optimizer.zero_grad()
		temporal_difference_loss.backward(retain_variables = True)

		self.optimizer.step()
	
	def update(self, reward, new_signal):
		new_state = torch.Tensor(new_signal).float().unsqueeze(0)
		self.memory_of_events.push_to_memory((self.last_state, new_state, \
			torch.LongTensor([int(self.last_action)]), \
				torch.Tensor([self.last_reward])))
		
		action = self.select_action(new_state)

		if len(self.memory_of_events.memory_of_events) > 100:
			batch_state, batch_state_next, batch_reward, batch_action = \
				self.memory_of_events.get_sample(100)
			self.learn(batch_state, batch_state_next, batch_reward, \
				batch_action)
			
		self.last_action = action
		self.last_state = new_state
		self.last_reward = reward
		self.reward_window.append(reward)

		if len(self.reward_window) > 1000:
			del self.reward_window[0]
		
		return action
	
	def score(self):
		return sum(self.reward_window) / (len(self.reward_window) + 1.)
	
	def save(self):
		torch.save({'state_dict': self.neural_network_model.state_dict(), \
			'optimizer': self.optimizer.state_dict()}, 'last_brain.pth')
	
	def load(self):

		if os.path.isfile('last_brain.pth'):
			print('==> loading checkpoint...')
			
			checkpoint = torch.load('last_brain.pth')

			self.neural_network_model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])

			print('done!')
		
		else:
			print('no checkpoint found...')


