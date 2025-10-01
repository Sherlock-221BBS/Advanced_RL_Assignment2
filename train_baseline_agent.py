#!/usr/bin/env python3
"""
Reference Training Script for PettingZoo Pong

This script serves as a COMPLETE REFERENCE for students on how to train agents.
Students should study this code to understand:
- How to set up training environments
- How to implement training loops
- How to use experience replay
- How to save/load models
- How to track training progress

Students can modify this script to train their own agents.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from pettingzoo_pong_wrapper import PettingZooPongWrapper
from baseline_agent import BaselineAgent, BaselineQNetwork

def train_baseline_agent(num_episodes=1000, save_path="baseline_agent.pt"):
    """
    REFERENCE TRAINING SCRIPT for students.

    This script demonstrates how to:
    - Create and configure the PettingZoo environment
    - Initialize and train a baseline agent
    - Use the agent's built-in update() method
    - Track training progress
    - Save the trained model

    Students can use this as a template for training their own agents.

    Args:
        num_episodes: Number of episodes to train for
        save_path: Path to save the trained model
    """
    print("Reference Training Implementation")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print(f"Save path: {save_path}")
    print()

    # STEP 1: Create the PettingZoo environment
    # Students should use the same environment wrapper
    env = PettingZooPongWrapper()

    # STEP 2: Create baseline agent (contains all hyperparameters and training components)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create action space for agent
    from gymnasium import spaces
    action_space = spaces.Discrete(6)

    print("Creating baseline agent...")
    # Create agent (contains all networks, optimizer, memory, hyperparameters, etc.)
    agent = BaselineAgent(agent_id=0, action_space=action_space)
    agent.reset_epsilon()  # Start with fresh epsilon
    print("Agent created successfully!")

    # STEP 5: Training statistics tracking
    episode_rewards = []
    episode_lengths = []
    epsilon_values = []

    print("Starting training...")
    print("Episode   Epsilon    Avg Reward   Avg Length")
    print("-" * 45)

    for episode in range(num_episodes):
        # STEP 6: Reset environment for new episode
        obs, _ = env.reset()
        done = False
        total_reward = [0, 0]
        steps = 0

        # STEP 7: Episode loop - collect experience
        while not done and steps < 1000:  # Limit episode length
            # Get actions for both agents
            action0 = agent.act(obs[0])  # Use agent's act method with agent's epsilon
            action1 = simple_ai_action(obs[1])  # Simple opponent for training

            actions = [action0, action1]

            # STEP 8: Execute actions in environment
            next_obs, rewards, done, _, _ = env.step(actions)

            # STEP 9: Update agent (agent handles its own training internally)
            agent.update(obs[0], action0, rewards[0], next_obs[0], done)

            total_reward[0] += rewards[0]
            obs = next_obs
            steps += 1

            # Debug: Print every 100 steps to see progress
            if steps % 100 == 0:
                print(f"    Step {steps}: Reward so far: {total_reward[0]:.1f}")

        # Update agent's epsilon
        agent.update_epsilon()

        # Track statistics
        episode_rewards.append(total_reward[0])
        episode_lengths.append(steps)
        epsilon_values.append(agent.get_epsilon())

        # Print progress every episode for better feedback
        if (episode + 1) % 1 == 0:
            current_epsilon = agent.get_epsilon()
            print(f"Episode {episode + 1:<3}: Reward: {total_reward[0]:<8.1f} Epsilon: {current_epsilon:.3f} Steps: {steps:<4}")

    # Save trained model using agent's save method
    agent.save_model(save_path)
    print(f"\nTraining completed! Model saved to {save_path}")

    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths[-100:]):.1f}")
    print(f"Final epsilon: {agent.get_epsilon():.3f}")

    # Close environment
    env.close()

    return agent

def simple_ai_action(obs):
    """
    Simple opponent AI for training.

    This provides a basic opponent for the agent to train against.
    Students can replace this with more sophisticated opponents.
    """
    # Random action with bias toward center
    return random.choices([0, 1, 2, 3, 4, 5], weights=[0.3, 0.1, 0.2, 0.2, 0.1, 0.1])[0]

if __name__ == "__main__":
    # Train baseline agent (reduced episodes for faster testing)
    print("Starting baseline agent training...")
    trained_network = train_baseline_agent(num_episodes=10)  # Very short training for testing

    print("\nBaseline agent training completed!")
    print("You can now use this agent to evaluate student submissions.")