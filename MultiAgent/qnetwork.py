# -*- coding: utf-8 -*-
"""
Created on March 30 13:14:56 2019

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
    
    def __init__(self, state_space, action_space, seed, hidden_layer1 = 512, hidden_layer2 = 256):
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
        
        self.norm0 = nn.BatchNorm1d(state_space)
        self.norm1 = nn.BatchNorm1d(hidden_layer1)
        self.norm2 = nn.BatchNorm1d(hidden_layer2) 
        
        self.reset_parameters()
    
    
    def hidden_init(self, layer):
        
        n = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(n)
        
        return (-lim, lim)
    
    
    def reset_parameters(self):        
        
        self.layer1.weight.data.uniform_(*self.hidden_init(self.layer1))
        self.layer2.weight.data.uniform_(*self.hidden_init(self.layer2))
        self.layer3.weight.data.uniform_(-3e-1, 3e-1)
    
    
    def forward(self, state):
        '''
        Get the action map from state map
        '''
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        
        norm0   = self.norm0(state)
        output1 = F.relu(self.norm1(self.layer1(norm0)))
        output2 = F.relu(self.norm2(self.layer2(output1)))
        output3 = F.tanh(self.layer3(output2))
        
        return output3
 

class Critic(nn.Module):
    '''
    Critic Q network
    '''
    
    def __init__(self, state_space, action_space, seed, hidden_layer1 = 512, hidden_layer2 = 256):
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
        
        self.norm0 = nn.BatchNorm1d(state_space)
        
        self.reset_parameters()
       
        
    def hidden_init(self, layer):
        
        n = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(n)
        
        return (-lim, lim)
    
    
    def reset_parameters(self):        
        
        self.layer1.weight.data.uniform_(*self.hidden_init(self.layer1))
        self.layer2.weight.data.uniform_(*self.hidden_init(self.layer2))
        self.layer3.weight.data.uniform_(-3e-3, 3e-3)
        
        
    def forward(self, state, action):
        '''
        Get the action map from state map
        '''
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        
        norm0   = self.norm0(state)
        output1 = F.relu(self.layer1(norm0))
        output2 = torch.cat((output1, action), dim = 1)
        output3 = F.relu(self.layer2(output2))        
        output4 = self.layer3(output3)
        
        return output4
    
        
class DDPGAgent:
    '''
    Deep Q Network Learning Agent
    '''
    
    def __init__(self, state_space, action_space, seed, device, mem_buffer = int(1e7), mem_batch = 256, gamma = 0.99, tau = 1e-3, learning_rate = 5e-3, update_freq = 16, weight_decay = 0.):
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
        
        state = torch.from_numpy(state).float().to(self.device)
        #state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
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
        
        
    def reset(self):
        '''
        Reset state
        '''
        self.noise.reset()
        
        
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
    
    
class MDDPGAgent:
    '''
    Multi Agent
    '''
    
    def __init__(self, config:dict):
        '''
        Initialize Multi agent operator
        
        params config       : agent's network parameters with following keys and defaults.
        
             num_agents     : number of agents interacting with environment in parallel
             state_space    : state space of the environment
             action_space   : available action space for agent 
             mem_buffer     : memory buffer for experience replay
             mem_batch      : batch size for experience replay
             gamma          : discount factor
             tau            : soft update for target weights update
             learning_rate  : learning rate of the network
             update_freq    : update frequency for target weights
        '''
        
        self.num_agents     = config.get('num_agents')
        self.state_space    = config.get('state_space')
        self.action_space   = config.get('action_space')
        self.seed           = config.get('seed')
        self.device         = config.get('device')
        self.mem_buffer     = config.get('mem_buffer', int(1e7))
        self.mem_batch      = config.get('mem_batch', 256)
        self.gamma          = config.get('gamma', 0.99)
        self.tau            = config.get('tau', 1e-3)
        self.learning_rate  = config.get('learning_rate', 5e-3)
        self.update_freq    = config.get('update_freq', 16)
        self.update_count   = 0
        self.weight_decay   = config.get('weight_decay', 0)
        
        self.agents = list()
        
        for i in range(self.num_agents):
            agent = DDPGAgent(self.state_space, self.action_space, self.seed, self.device,
                              mem_buffer = self.mem_buffer, mem_batch = self.mem_batch, 
                              gamma = self.gamma, tau = self.tau, learning_rate = self.learning_rate, 
                              update_freq = self.update_freq, weight_decay = self.weight_decay)
            
            self.agents.append(agent)
            
        
    def select_actions(self, states, epsilon, training = True):
        '''
        Select optimal actions for each agent's states using epsilon greedy policy
        param state: current states of all active agents
        epsilon: epsilon weight for greedy action
        return: agents' actions for their states
        '''
        
        actions = list()
        
        for agent, state in zip(self.agents, states): 
            local_state = state
            #local_state = np.expand_dims(state, axis=0)
            action = agent.select_action(local_state, epsilon, training = training)
            actions.append(action)
        
        return actions
        
        
    def experience(self, states, actions, rewards, new_states, terminal_states, share = False):
        '''
        Record the environment action experience and update Q-values of agents' dqn
        param states: previous states of the environment for each agent
        param actions: actions taken by agents in their previous states
        param rewards: rewards received from environment from previous state-action of agents
        param new_states: transitioned states for each agent
        param terminal_states: whether new states are a terminal state or not for each agent
        param share : share the experience among agents
        '''
        
        if share:
            for agent in self.agents:
                for i in range(self.num_agents):
                    agent.experience(states[i], actions[i], rewards[i], new_states[i], terminal_states[i])
        else: 
            for i in range(self.num_agents):
                agent = self.agents[i]
                agent.experience(states[i], actions[i], rewards[i], new_states[i], terminal_states[i])
    
    def reset(self):
        '''
        Reset agents' state
        '''
        
        for agent in self.agents:
            agent.reset()
        
        
    def save_memory(self, checkpoint_name_critic, checkpoint_name_actor):
        '''
        Save current learned parameters of the DQN for all agents
        '''
        
        critic_names = checkpoint_name_critic.split('.')
        actor_names  = checkpoint_name_actor.split('.')
        
        for i in range(self.num_agents):
            
            agent = self.agents[i]
            
            critic_name = critic_names[0] + str(i) + '.' + critic_names[1]
            actor_name  = actor_names[0] + str(i) + '.' + actor_names[1]
            
            agent.save_memory(critic_name, actor_name)
        
        
    def restore_memory(self, checkpoint_name_critic, checkpoint_name_actor):
        '''
        Restore learned parameters of DQN for all agents
        '''
        
        critic_names = checkpoint_name_critic.split('.')
        actor_names  = checkpoint_name_actor.split('.')
        
        for i in range(self.num_agents):
            
            agent = self.agents[i]
            
            critic_name = critic_names[0] + str(i) + '.' + critic_names[1]
            actor_name  = actor_names[0] + str(i) + '.' + actor_names[1]
            
            agent.restore_memory(critic_name, actor_name)
        