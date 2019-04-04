# Navigation using Deep Reinforcement Learning Agent

In this project, an agent has been trained using Reinforcement Learning algorithm to navigate an environment that has yellow and blue banans. The task for agent is to collect yellow bananas (as many as possible) and avoid blue bananas.


## The Environment

Unity's framework's Banana Navigator environment has been used to simulate the environment. State space of the environment has 37 dimensions and contains agent's velocity, along with ray based perception of the objects around agent's forward direction.

![environment](images/environment.gif)


## The Agent

An agent's task is to navigate the Banana world and collect yellow bananas (as many as possible) while avoiding blue bananas. An agent has four possible actions available: 

- Move Forward
- Move Backward
- Turn Left 
- Turn Right

Each navigation is episodic (i.e. task will terminate at certain steps) and a trained agent is expected to achieve an average score of +13. 


## Getting Started

Make sure you have `python 3.6` installed under your virtual environment, along with Unity framework. To install dependencies, open the 'Navigation.ipynb' notebook and run first cell.

!pip -q install ./python

## Code Structure

To start running the project, open and run the notebook 'Navigation.ipynb'. Structure of the code as is below:

1. qnetwork.py: this file implemented the DQN agent.

2. navigator.py: training of the agent is controlled from this file

3. Navigation.ipynb: notebook has trained network and it's results.
 
4. 'checkpoint.pth': trained network's parameters. 

Agent can restore it's memory using navigator.restore_memoery function.
