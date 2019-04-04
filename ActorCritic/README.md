# Continous Control Operation using Deep Reinforcement Learning Agent

In this project, an agent has been trained on a continous space operation using Reinforcement Learning algorithm to hold the moving target using a double-joined arm. The goal of the agent is to maintain the double-joined arm location at the target location, for as many time steps as possible.


## The Environment

Unity's framework's Reacher environment has been used to simulate the environment. The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints.

![environment](images/environment.gif)


## The Agent

An agent's task is to maintain the double-joined arm location at the target location, for as many time steps as possible. 

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 


## Getting Started

Make sure you have `python 3.6` installed under your virtual environment, along with Unity framework. To install dependencies, open the 'Continuous_Control.ipynb' notebook and run first cell.

!pip -q install ./python

## Code Structure

To start running the project, open and run the notebook 'Continuous_Control.ipynb'. Structure of the code as is below:

1. qnetwork.py: this file implemented the agent using Deep Deterministic Policy Gradeint (DDPG) algorithm.

2. navigator.py: training of the agent is controlled from this file

3. Continuous_Control.ipynb: notebook has trained network and it's results.
 
4. 'checkpoint_actor.pth': actor's trained network's parameters.

5. 'checkpoint_critic.pth': critic's trained network's parameters.
 
Agent can restore it's memory using navigator.restore_memoery function.

