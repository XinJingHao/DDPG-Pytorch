import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, maxaction):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, 300)
		self.l3 = nn.Linear(300, action_dim)

		self.maxaction = maxaction

	def forward(self, state):
		a = torch.relu(self.l1(state))
		a = torch.relu(self.l2(a))
		a = torch.tanh(self.l3(a)) * self.maxaction
		return a


class Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(Q_Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, net_width)
		self.l2 = nn.Linear(net_width, 300)
		self.l3 = nn.Linear(300, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q = F.relu(self.l1(sa))
		q = F.relu(self.l2(q))
		q = self.l3(q)

		return q




class DDPG_Agent(object):
	def __init__(self, opt):
		self.actor = Actor(opt.state_dim, opt.action_dim, opt.net_width, opt.max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=opt.a_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.q_critic = Q_Critic(opt.state_dim, opt.action_dim, opt.net_width).to(device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=opt.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.replay_buffer = ReplayBuffer(opt.state_dim, opt.action_dim, int(5e5))

		self.max_action = opt.max_action
		self.batch_size = opt.batch_size
		self.gamma = opt.gamma
		self.tau = 0.005
		
	def select_action(self, state):
		#only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			a = self.actor(state)
		return a.cpu().numpy().flatten()

	def train(self):
		# Compute the target Q
		with torch.no_grad():
			s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
			target_a_next = self.actor_target(s_next)
			target_Q= self.q_critic_target(s_next, target_a_next)
			target_Q = r + (1 - dw) * self.gamma * target_Q  #dw: die or win

		# Get current Q estimates
		current_Q = self.q_critic(s, a)

		# Compute critic loss
		q_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the q_critic
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		# Update the Actor
		a_loss = -self.q_critic(s,self.actor(s)).mean()
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()

		with torch.no_grad():
			# Update the frozen target models
			for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self,EnvName,episode):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,int(episode/1e3)))
		torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,int(episode/1e3)))


	def load(self,EnvName,episode):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName,episode)))
		self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName,episode)))


class ReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.reward = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, state_dim))
		self.dw = np.zeros((max_size, 1))

		self.device = device


	def add(self, state, action, reward, next_state, dw):
		#每次只放入一个时刻的数据
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.dw[self.ptr] = dw #0,0,0，...，1

		self.ptr = (self.ptr + 1) % self.max_size #存满了又重头开始存
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.dw[ind]).to(self.device)
		)



