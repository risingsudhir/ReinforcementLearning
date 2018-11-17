# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:25:36 2018

@author: tiwarisu
"""

import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
                        
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        # epsilon threshold
        self.min_epsilon = 0.000000001
        # keep total steps during environment interation during its lifetime
        self.steps = 0
        # starting epsilon value
        self.epsilon = 0.99
        # decay threshold
        self.epsilon_decay = 0.98
        # starting alpha value
        self.alpha = 0.067
        self.gamma = 1.
        
        # 9.2- epsilon_decay - 0.99, alpha = 0.025, steps = 50
        
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        # assign euqal probability to each available action
        probs = np.ones(self.nA, dtype = float) / self.nA

        if (state in self.Q):
            
            # inject bias for optimal action
            probs *= self.epsilon
            
            optim_action = np.argmax(self.Q[state])
            
            probs[optim_action] = (1. - self.epsilon) + self.epsilon / self.nA
                               
        return np.random.choice(self.nA, p = probs)

        
    def experience(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        self.steps += 1
        
        if done: 
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
        else:
            
            if self.steps > 20000:                
                
                # get the estimate by finding the expected value from greedy selectoin
                optim_action = self.get_optim_action(next_state)
                
                # update Q table with new estimates
                self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][optim_action] - self.Q[state][action])
                
            else:
                
                # get the estimated reward by estimated expected reward of the state with weighted rewards for all actions
                expected_value = self.get_expected_state_value(next_state)
                
                # update Q table with new estimate
                self.Q[state][action] += self.alpha * (reward + self.gamma * expected_value - self.Q[state][action])
                                                       
        
        # update epsilon with every 100th step
        if self.steps % 100 == 0:
            self.epsilon = max(self.epsilon * (self.epsilon_decay), self.min_epsilon)
                        
                  
    def get_optim_action(self, state):
        '''
        Returns most greedy action for the given state
        '''
        
        return np.argmax(self.Q[state])        
        
        
    def get_expected_state_value(self, state):
        '''
        Get the expected state value for the action
        '''
        
        # assign equal probabilities for each action
        probs = np.ones(self.nA, dtype = float) / self.nA
        
        if state in self.Q:
            
            # inject probability bias for most greedy action
            probs *= self.epsilon
            
            optim_action = np.argmax(self.Q[state])
            
            probs[optim_action] = (1. - self.epsilon) + (self.epsilon / self.nA)
            
        return np.sum(self.Q[state] * probs)
            
            
            
        