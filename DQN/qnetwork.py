# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:36:17 2019

@author: tiwarisudhir
"""

import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

# set device placement
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    '''
    Q network for state and action space
    '''
    
    def __init__(self, state_space, action_space, seed, hidden_layer1 = 64 , hidden_layer2 = 64):
        '''
        Initialize Q network
        param state_space: state space size 
        param action_space: action space size 
        param seed: initial seed
        param hidden_layer1: first hidden layer neurons 
        param hidden_layer2: second hidden layer neuron
        '''
        
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.layer1 = nn.Linear(state_space, hidden_layer1)
        self.layer2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.layer3 = nn.Linear(hidden_layer2, action_space)
        
    
    def forward(self, state):
        '''
        Get the action map from state map
        '''
        
        output = F.relu(self.layer1(state))
        output = F.relu(self.layer2(output))
        
        return self.layer3(output)
            
    
        
class DQNAgent:
    '''
    Deep Q Network Learning Agent
    '''
    
    def __init__(self, state_space, action_space, seed, device, mem_buffer = int(1e5), mem_batch = 64, gamma = 0.99, tau = 1e-3, learning_rate = 5e-4, update_freq = 4):
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
        
        # define on-policy and off-policy dqns
        self.dqn_off_policy = QNetwork(self.state, self.action, seed)
        self.dqn_on_policy  = QNetwork(self.state, self.action, seed)
        
        # optimizer for on-policy dqn
        self.optimizer = optim.Adam(self.dqn_on_policy.parameters(), lr = learning_rate)
        
        self.memory = ExperienceReplay(mem_buffer, mem_batch, seed)
        
    
    def select_action(self, state, epsilon, training = True):
        '''
        Select optimal action for state using epsilon greedy policy
        param state: current state of the agent
        epsilon: epsilon weight for greedy action
        return: agent's action for given state
        '''
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.dqn_on_policy.eval()
        
        with torch.no_grad():
            actions = self.dqn_on_policy(state)
        
        self.dqn_on_policy.train()
        
        # select actions
        actions = actions.cpu().data.numpy()
                        
        if training == False: 
            return np.argmax(actions)
            
        # assign euqal probability to each available action
        probs = np.ones(self.action, dtype = float) / self.action

        if epsilon > 0.:
            # inject bias for optimal action
            probs *= epsilon
            optim_action = np.argmax(actions)
            probs[optim_action] = (1. - epsilon) + epsilon / self.action
                               
        return np.random.choice(self.action, p = probs)
        
        
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
        
        # get q values from off policy network
        Q_target_updated = self.dqn_off_policy(new_states).detach().max(1)[0].unsqueeze(1)
        
        # do not update rewards for terminal states
        Q_target = rewards + self.gamma * Q_target_updated * (1 - terminals)
        
        Q_expected = self.dqn_on_policy(states).gather(1, actions)
        
        # minimize losses in expected Q-Values and Target Q-Values
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
                
        
        # update off-policy network parameters from on-policy network
        self.update_target()
                
        
    def update_target(self):
        '''
        Update target network parameters from local (on-policy) network parameters
        '''
        
        for off_weights, on_weights in zip(self.dqn_off_policy.parameters(), self.dqn_on_policy.parameters()):
            
            updated_weights = self.tau * on_weights + (1. - self.tau) * off_weights
            
            off_weights.data.copy_(updated_weights)
        
                 
    def save_memory(self, checkpoint_name):
        '''
        Save current learned parameters of the DQN
        '''
        
        torch.save(self.dqn_on_policy.state_dict(), checkpoint_name)
        
    def restore_memory(self, checkpoint_name):
        '''
        Restore learned parameters of DQN
        '''
        
        self.dqn_on_policy.load_state_dict(torch.load(checkpoint_name))
        self.dqn_off_policy.load_state_dict(torch.load(checkpoint_name))
        
        
        
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
        actions     = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards     = torch.from_numpy(np.vstack(rewards)).float().to(device)
        new_states  = torch.from_numpy(np.vstack(new_states)).float().to(device)
        terminals   = torch.from_numpy(np.vstack(terminals).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, new_states, terminals)
                                                           
        
        
        
            
        
        
        
    
    
    
    