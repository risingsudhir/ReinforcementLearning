# -*- coding: utf-8 -*-
"""
Created on March 2 13:49:56 2019

@author: tiwarisudhir
"""

import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

# set device placement
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    '''
    Actor Q network
    '''
    
    def __init__(self, state_space, action_space, seed, hidden_layer1 = 64 , hidden_layer2 = 64):
        '''
        Initialize Actor Q network
        param state_space: state space size 
        param action_space: action space size 
        param seed: initial seed
        param hidden_layer1: first hidden layer neurons 
        param hidden_layer2: second hidden layer neuron
        '''
        
        super(Actor, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.layer1 = nn.Linear(state_space, hidden_layer1)
        self.layer2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.layer3 = nn.Linear(hidden_layer2, action_space)
        
        self.norm1 = nn.BatchNorm1d(hidden_layer1)
    
    def forward(self, state):
        '''
        Get the action map from state map
        '''
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        
        output1 = F.relu(self.layer1(state))
        output2 = self.norm1(output1)
        output3 = F.relu(self.layer2(output2))
        output4 = F.tanh(self.layer3(output3))
        
        return output4
 

class Critic(nn.Module):
    '''
    Critic Q network
    '''
    
    def __init__(self, state_space, action_space, seed, hidden_layer1 = 64 , hidden_layer2 = 64):
        '''
        Initialize critic Q network
        param state_space: state space size 
        param action_space: action space size 
        param seed: initial seed
        param hidden_layer1: first hidden layer neurons 
        param hidden_layer2: second hidden layer neuron
        '''
        
        super(Critic, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.layer1 = nn.Linear(state_space, hidden_layer1)
        self.layer2 = nn.Linear(hidden_layer1 + action_space, hidden_layer2)
        self.layer3 = nn.Linear(hidden_layer2, action_space)
        
        self.norm1 = nn.BatchNorm1d(hidden_layer1)
        
        
    def forward(self, state, action):
        '''
        Get the action map from state map
        '''
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        
        output1 = F.relu(self.layer1(state))
        output2 = self.norm1(output1)
        
        output3 = torch.cat((output2, action), dim = 1)
        
        output4 = F.relu(self.layer2(output3))        
        output5 = self.layer3(output4)
        
        return output5
    
        
class DDPGAgent:
    '''
    Deep Q Network Learning Agent
    '''
    
    def __init__(self, state_space, action_space, seed, device, mem_buffer = int(1e5), mem_batch = 64, gamma = 0.99, tau = 1e-3, learning_rate = 5e-4, update_freq = 4, weight_decay = 0.):
        '''
        Initialize agent
        param state_space   : state space of the environment
        param action_space  : available action space for agent 
        param mem_buffer    : memory buffer for experience replay
        param mem_batch     : batch size for experience replay
        param gamma         : discount factor
        param tau           : soft update for target weights update
        param learning_rate : learning rate of the network
        param update_freq   : update frequency for target weights
        '''
        
        self.state  = state_space
        self.action = action_space
        self.seed   = random.seed(seed)
        self.device = device
        
        self.memory_buffer  = mem_buffer 
        self.memory_batch   = mem_batch
        self.gamma          = gamma
        self.tau            = tau
        self.learning_rate  = learning_rate
        self.update_freq    = update_freq
        self.update_count   = 0
        
        # define on-policy and off-policy actors
        self.actor_off_policy = Actor(self.state, self.action, seed).to(self.device)
        self.actor_on_policy  = Actor(self.state, self.action, seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_on_policy.parameters(), lr = learning_rate)
        
         # define on-policy and off-policy critic
        self.critic_off_policy = Critic(self.state, self.action, seed).to(self.device)
        self.critic_on_policy  = Critic(self.state, self.action, seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_on_policy.parameters(), lr = learning_rate, weight_decay = weight_decay)
        
        self.update_target(self.critic_off_policy, self.critic_on_policy, force = True)
        self.update_target(self.actor_off_policy, self.actor_on_policy, force = True)
        
        self.memory = ExperienceReplay(mem_buffer, mem_batch, seed)
        self.noise = Noise(action_space, seed) 
        
    
    def select_action(self, state, epsilon, training = True):
        '''
        Select optimal action for state using epsilon greedy policy
        param state: current state of the agent
        epsilon: epsilon weight for greedy action
        return: agent's action for given state
        '''
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.actor_on_policy.eval()
        
        with torch.no_grad():
            actions = self.actor_on_policy(state)
        
        self.actor_on_policy.train()
        
        # select actions
        actions = actions.cpu().data.numpy()
        
        if training:
            actions += self.noise.sample()
        
        return np.clip(actions, -1., 1.)
        
        
    def experience(self, state, action, reward, new_state, terminal_state):
        '''
        Record the environment action experience and update Q-values of the dqn
        param state: previous state of the environment
        param action: action taken by agent in previous state
        param reward: reward received from environment from previous state-action
        param new_state: transitioned state
        param terminal_state: whether new state is a terminal state or not
        '''
        
        # add experience to memory
        self.memory.add(state, action, reward, new_state, terminal_state)
        self.update_count += 1
        
        self.update_count = (self.update_count + 1) % self.update_freq

        if (self.update_count == 0) and self.memory.count() > self.memory_batch:
            
            # train dqn from experience samples
            samples = self.memory.select_samples(self.device)
            
            self.update(samples)
            
            
    def update(self, experiences):
        '''
        Update Q-Value estimates by using past experience tuples
        param experiences: tuples of <state, action, reward, new_state, terminal>
        '''
        
        states, actions, rewards, new_states, terminals = experiences
        
        # get actors actions for new states
        new_actions = self.actor_off_policy(new_states)
        
        # get Q values from critic off policy network
        Q_target_updated = self.critic_off_policy(new_states, new_actions)
        
        # update Q target
        Q_target = rewards + self.gamma * Q_target_updated * (1 - terminals)
        
        # get critics' Q values for previous states and actions
        Q_expected = self.critic_on_policy(states, actions)
        
        # minimize critic expected loss
        critic_loss = F.mse_loss(Q_expected, Q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_on_policy.parameters(), 1)
        self.critic_optimizer.step()
        
        # minimize actor loss
        actions_actor = self.actor_on_policy(states)
        actor_loss = -self.critic_on_policy(states, actions_actor).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
                
        # update off-policy network parameters from on-policy network
        self.update_target(self.critic_off_policy, self.critic_on_policy)
        self.update_target(self.actor_off_policy, self.actor_on_policy)
                
        
    def update_target(self, off_policy, on_policy, force = False):
        '''
        Update target network parameters from local (on-policy) network parameters
        '''
        
        for off_weights, on_weights in zip(off_policy.parameters(), on_policy.parameters()):
            
            if force:
                updated_weights = off_weights
            else:
                updated_weights = self.tau * on_weights + (1. - self.tau) * off_weights
            
            off_weights.data.copy_(updated_weights)
        
                 
    def save_memory(self, checkpoint_name_critic, checkpoint_name_actor):
        '''
        Save current learned parameters of the DQN
        '''
        
        torch.save(self.critic_on_policy.state_dict(), checkpoint_name_critic)
        torch.save(self.actor_on_policy.state_dict(), checkpoint_name_actor)
        
    def restore_memory(self, checkpoint_name_critic, checkpoint_name_actor):
        '''
        Restore learned parameters of DQN
        '''
        
        self.critic_on_policy.load_state_dict(torch.load(checkpoint_name_critic))
        self.critic_off_policy.load_state_dict(torch.load(checkpoint_name_critic))
        
        self.actor_on_policy.load_state_dict(torch.load(checkpoint_name_actor))
        self.actor_off_policy.load_state_dict(torch.load(checkpoint_name_actor))
        
        
class ExperienceReplay:
    '''
    Experience Replay to store past experiences
    '''
    
    def __init__(self, mem_buffer, mem_batch, seed):
        '''
        Initialize experience memory buffer
        '''
                
        self.mem_batch = mem_batch
        self.seed = random.seed(seed)
        
        # memory buffer to store experience
        self.memory = deque(maxlen = mem_buffer)
        self.experience = namedtuple(typename = 'experience', field_names = ['state', 'action', 'reward', 'new_state', 'terminal'])
        
        
    def add(self, state, action, reward, new_state, done):
        '''
        Record current experience
        '''
        
        experience = self.experience(state = state, action = action, reward = reward, new_state = new_state, terminal = done)
        self.memory.append(experience)
        
    def count(self):
        '''
        Returns total recorded experiences in the buffer
        '''
        
        return len(self.memory)
        
    def select_samples(self, device):
        '''
        Select experience samples from previous experience and place on given device
        '''
        
        samples = random.sample(self.memory, k = self.mem_batch)
        
        states      = [sample.state for sample in samples if sample is not None]
        actions     = [sample.action for sample in samples if sample is not None]
        rewards     = [sample.reward for sample in samples if sample is not None]
        new_states  = [sample.new_state for sample in samples if sample is not None]
        terminals   = [sample.terminal for sample in samples if sample is not None]

        states      = torch.from_numpy(np.vstack(states)).float().to(device)
        actions     = torch.from_numpy(np.vstack(actions)).float().to(device)
        rewards     = torch.from_numpy(np.vstack(rewards)).float().to(device)
        new_states  = torch.from_numpy(np.vstack(new_states)).float().to(device)
        terminals   = torch.from_numpy(np.vstack(terminals).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, new_states, terminals)
    
    
class Noise:
    '''
    Action space noise
    '''
    
    def __init__(self, action_space, seed, mu = 0., theta = 0.15, sigma = 0.1):
        '''
        Initialize experience memory buffer
        '''
        
        self.mu = mu * np.ones(action_space)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        
        self.reset()
                
    def reset(self):
        '''
        Reset mean state
        '''
        self.state = copy.copy(self.mu)
                
    def sample(self):
        '''
        Update state and return noise sample
        '''
        
        noise = np.array([random.random() for i in range(len(self.state))])
        delta = self.theta * (self.mu - self.state) + self.sigma * noise
        
        self.state += delta
        
        return self.state
        