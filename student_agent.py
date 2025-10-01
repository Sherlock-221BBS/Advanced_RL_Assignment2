import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    """
    Deep Q-Network for Pong agent.
    Input: RGB image observation (210, 160, 3) - Students must handle preprocessing
    Output: 6 Q-values (one for each action: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=FIRE+RIGHT, 5=FIRE+LEFT)

    REQUIRED ARCHITECTURE: Students must implement this network
    """
    def __init__(self, input_size=None, hidden_size=128, output_size=6):
        super(QNetwork, self).__init__()
        # TODO: Implement your Q-network architecture
        # You need to handle RGB input (210, 160, 3) -> 6D output
        # You can modify hidden_size, add CNN layers, etc.
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass

class StudentAgent:
    """
    Multi-Agent RL Agent for PettingZoo Pong

    This agent works with the PettingZoo Pong environment.
    You must implement your multi-agent RL algorithm here.

    INPUT: RGB image observations (210, 160, 3)
    OUTPUT: 6 actions (0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=FIRE+RIGHT, 5=FIRE+LEFT)

    REQUIRED INTERFACE FUNCTIONS:
    - __init__(self, agent_id, action_space): Initialize your agent
    - act(self, obs): Select action during evaluation (DO NOT CHANGE THIS FUNCTION)
    - load_model(self, model_data): Load your trained model
    - save_model(self, filepath): Save your trained model
    - update(self, obs, action, reward, next_obs, done): Update your agent (for training)

    You can use any multi-agent RL algorithm taught in class.
    """
    def __init__(self, agent_id, action_space):
        self.agent_id = agent_id
        self.action_space = action_space
        self.n_actions = 6  # PettingZoo Pong has 6 actions

        # TODO: Initialize your multi-agent RL algorithm here
        # Examples: DQN, PPO, MADDPG, etc.
        # You can add CNN layers to process RGB observations
        pass

    def act(self, obs):
        """
        Select an action using your policy. DO NOT CHANGE THE NAME OF THE FUNCTION.

        Args:
            obs: RGB image observation of shape (210, 160, 3) from PettingZoo

        Returns:
            action: An integer (0-5) representing the action to take
        """
        # TODO: Implement your action selection logic
        # This function will be used during evaluation
        # Make sure it returns a valid action (0-5)
        return 0  # Placeholder - replace with your implementation

    def update(self, obs, action, reward, next_obs, done):
        """
        Update your agent using a transition.

        Args:
            obs: Current observation (RGB image)
            action: Action taken
            reward: Reward received
            next_obs: Next observation (RGB image)
            done: Whether episode is finished
        """
        # TODO: Implement your learning update
        # This function will be called during training
        pass

    def save_model(self, filepath):
        """
        Save your trained model.

        Args:
            filepath: Path to save the model
        """
        # TODO: Implement model saving
        print(f"Model saved to {filepath}")

    def load_model(self, model_data):
        """
        Load your trained model.

        Args:
            model_data: The loaded model data
        """
        # TODO: Implement model loading
        print("Model loaded successfully")
