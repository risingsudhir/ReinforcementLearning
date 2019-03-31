# -*- coding: utf-8 -*-
"""
Created on March 30 15:03:06 2019

@author: tiwarisudhir
"""

import gym
import torch
import numpy as np
from collections import deque
from unityagents import UnityEnvironment
import qnetwork

# set device placement

SEED              = 0
MEM_BUFFER        = int(1e7)
MEM_BATCH         = 64
GAMMA             = 0.99
TAU               = 5e-3
LEARNING_RATE     = 7e-4
UPDATE_FREQ       = 8
CHECKPOINT_CRITIC = 'checkpoint_critic.pth'
CHECKPOINT_ACTOR  = 'checkpoint_actor.pth'
MIN_THRESHOLD     = 0.5
MAX_THRESHOLD     = 1.0
device            = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_environment():
    '''
    Get gym environment
    '''
    
    env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis")
    
    return env
    

def get_agent(env):
    '''
    Get MDDPG agent
    '''
    
    brain_name = env.brain_names[0]
    brain      = env.brains[brain_name]
    env_info   = env.reset(train_mode=True)[brain_name]

    agents  = len(env_info.agents)
    states  = env_info.vector_observations.shape[1]
    actions = brain.vector_action_space_size
  
    config = dict()
    config['num_agents']    = agents
    config['state_space']   =  states
    config['action_space']  = actions
    config['seed']          = SEED
    config['device']        = device
    config['mem_buffer']    = MEM_BUFFER
    config['mem_batch']     = MEM_BATCH
    config['gamma']         = GAMMA
    config['tau']           = TAU
    config['learning_rate'] = LEARNING_RATE
    config['update_freq']   = UPDATE_FREQ
      
    agent = qnetwork.MDDPGAgent(config)

    return agent
    
    
def train_players(env, agent, episodes = 500, max_steps = 2000, share_experience = False, 
                  epsilon_max = 1., epsilon_min = 0.001, epsilon_decay = 0.999):
    '''
    Train the agent
    param episodes: number of episodes to train for
    param max_steps: max number of time steps per episode
    param share_experience: shre experience with agents or not
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
        
        #agent.reset()
        
        step       = 0
        scores     = np.zeros(agent.num_agents)
        brain_name = env.brain_names[0]
        env_info   = env.reset(train_mode = True)[brain_name]
        states     = env_info.vector_observations
        
        while step < max_steps:
            
            # get agents' action
            actions = agent.select_actions(states, epsilon)
            
            # take action that agents have selected
            env_info = env.step(actions)[brain_name]
            
            # get the transitioned state and reward
            new_states = env_info.vector_observations
            rewards    = env_info.rewards
            terminals  = env_info.local_done
               
            # pass the reward to agent
            agent.experience(states, actions, rewards, new_states, terminals, share_experience)
            
            # transition to new state
            step    += 1
            states  = new_states
            scores += rewards 
            
            if np.any(terminals):
                break
           
        # decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # record rewards
        reward = np.max(scores)
        min_reward = np.min(scores)
        episode_rewards.append(reward)
        window_rewards.append(reward)
        
        print('\r Episode-{}, Eps: {:.4f}, Reward: Avg {:.4f}'\
               .format((episode + 1), epsilon, np.mean(window_rewards)), end = "")
        
        if (episode + 1) % 100 == 0:
            print('\r Episode-{}, Eps: {:.4f}, Avg Reward: {:.4f}'.format((episode + 1), epsilon, np.mean(window_rewards)))
            
        if (np.mean(window_rewards) >= MIN_THRESHOLD) and (np.mean(window_rewards) > best_avg_rewards):
            
            if best_avg_rewards < MIN_THRESHOLD:
                print('\n Agents have learned to solve the task. Episode-{}, Avg Reward: {:.4f}'\
                      .format((episode + 1), np.mean(window_rewards)))
            
            best_avg_rewards = np.mean(window_rewards)
                  
            agent.save_memory(checkpoint_name_critic = CHECKPOINT_CRITIC, checkpoint_name_actor = CHECKPOINT_ACTOR)
            
            if best_avg_rewards >= MAX_THRESHOLD:
                print('\n Average reward beyond max expectation, stopping training now.')
                break
                              
    return episode_rewards
        
            
def restore_memory(agent):
    '''
    Restore agents' past learning
    '''
    
    agent.restore_memory(checkpoint_name_critic = CHECKPOINT_CRITIC, checkpoint_name_actor = CHECKPOINT_ACTOR)

    