# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:26:17 2018

@author: tiwarisu
"""

from agent import Agent
from monitor import interact
import gym
import numpy as np

env   = gym.make('Taxi-v2')
agent = Agent()

avg_rewards, best_avg_reward = interact(env, agent)
