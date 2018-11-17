# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:36:09 2018

@author: tiwarisu
"""

from collections import deque
import sys
import math
import numpy as np

def interact(env, agent, num_episodes=20000, window=100):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    
    # initialize best average reward
    best_avg_reward = -math.inf
    
    # initialize monitor for most recent rewards
    sample_rewards = deque(maxlen=window)
    
    # for each episode
    for i_episode in range(1, num_episodes+1):
        
        # reset current state to start a new episode
        action_reward = 0
        
        state = env.reset()
                
        while True:
            
            action = agent.select_action(state)
            
            next_state, reward, done, info = env.step(action)
            
            agent.experience(state, action, reward, next_state, done)
            
            # update the sampled reward
            action_reward += reward
            
            # update the state transition
            state = next_state
            
            if done:
                # save final sampled reward
                sample_rewards.append(action_reward)
                break
            
        if (i_episode >= 100):
            
            # get average reward from last 100 episodes
            avg_reward = np.mean(sample_rewards)
                        
            avg_rewards.append(avg_reward)
            
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        
        # monitor progress
        print("\rEpisode {}/{}, Steps: {}, Eps: {}, Alpha: {} || Best average reward {}".format(i_episode, num_episodes, agent.steps, agent.epsilon, agent.alpha, best_avg_reward), end="")
        
        sys.stdout.flush()
        
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
        
        if i_episode == num_episodes: print('\n')
        
    return avg_rewards, best_avg_reward
