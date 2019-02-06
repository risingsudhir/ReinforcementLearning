# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:49:56 2019

@author: tiwarisudhir
"""

import gym
import torch
import numpy as np
from collections import deque
from unityagents import UnityEnvironment
import qnetwork

# set device placement

SEED            = 0
MEM_BUFFER      = int(1e5)
MEM_BATCH       = 64
GAMMA           = 0.99
TAU             = 1e-3
LEARNING_RATE   = 5e-4
UPDATE_FREQ     = 4
CHECKPOINT_NAME = 'checkpoint.pth'
MIN_THRESHOLD   = 14. 
device          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_environment():
    '''
    Get gym environment and brain for the navigator
    '''
    
    env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
    
    return env
    

def get_agent(env):
    '''
    Get DQN agent and environment brain
    '''
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    states = len(env_info.vector_observations[0])
    actions = brain.vector_action_space_size
        
    agent = qnetwork.DQNAgent(states, actions, seed = SEED, device = device, 
                              mem_buffer = MEM_BUFFER, mem_batch = MEM_BATCH,
                              gamma = GAMMA, tau = TAU, 
                              learning_rate = LEARNING_RATE, update_freq = UPDATE_FREQ)

    return agent
    
    
def train_navigator(env, agent, episodes = 2000, max_steps = None, epsilon_max = 1., epsilon_min = 0.01, epsilon_decay = 0.997):
    '''
    Train a DQN for navigator to collect Bananas
    param episodes: number of episodes to train for
    param max_steps: max number of time steps per episode
    param epsilon_max: starting epsilon value for greedy policy selection
    param epsilon_end: min value of the epsilon for greedy policy selection
    param epsilon_decay: decay rate of epsilon per episode
    '''
    
    # monitor learning by keeping track of rewards
    epsilon             = epsilon_max
    best_avg_rewards    = 0.
    episode_rewards     = []
    window_rewards      = deque(maxlen = 100)
    
    for episode in range(episodes):
        
        step       = 0
        rewards    = 0.
        brain_name = env.brain_names[0]
        env_info   = env.reset(train_mode = True)[brain_name]
        state      = env_info.vector_observations[0]
        
        while (max_steps == None) or (step < max_steps):
            
            # get agent's action
            action = agent.select_action(state, epsilon)
            
            # take action that agent has selected
            env_info = env.step(action)[brain_name]
            
            # get the transitioned state and reward
            new_state = env_info.vector_observations[0]
            reward   = env_info.rewards[0]
            done     = env_info.local_done[0]
            
            # pass the reward to agent
            agent.experience(state, action, reward, new_state, done)
            
            # transition to new state
            state = new_state
            rewards += reward
            
            if done:
                break
                
            step += 1

        # decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # record rewards
        episode_rewards.append(rewards)
        window_rewards.append(rewards)
        
        print('\r Episode-{}, Eps: {:.4f}, Avg Reward: {:.2f}'.format((episode + 1), epsilon, np.mean(window_rewards)), end = "")
        
        if (episode + 1) % 100 == 0:
            print('\r Episode-{}, Avg Reward: {:.2f}'.format((episode + 1), np.mean(window_rewards)))
            
        if (np.mean(window_rewards) >= MIN_THRESHOLD) and (np.mean(window_rewards) > best_avg_rewards):
            
            best_avg_rewards = np.mean(window_rewards)
            
            print('\n Agent has learned to solve the task. Episode-{}, Avg Reward: {:.2f}'.format((episode + 1), best_avg_rewards))
                        
            agent.save_memory(checkpoint_name = CHECKPOINT_NAME)
                                
    return episode_rewards
        
            
def restore_memory(agent):
    '''
    Restore agent's past learning
    '''
    
    agent.restore_memory(checkpoint_name = CHECKPOINT_NAME)

    