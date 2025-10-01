# Multi-Agent Reinforcement Learning Pong Assignment

This repository contains materials for a multi-agent reinforcement learning assignment using the PettingZoo Pong environment.

## Assignment Overview

Students will implement a multi-agent reinforcement learning agent to play competitive Pong. The assignment focuses on:

- Processing RGB image observations (210, 160, 3)
- Implementing neural networks for visual reinforcement learning
- Training agents using any multi-agent RL algorithm
- Understanding competitive multi-agent scenarios

## Files for Students

The following files are provided for you to complete the assignment:

```
Assignment Files (Provided):
├── pettingzoo_pong_wrapper.py     # PettingZoo Pong environment wrapper
├── student_agent.py               # Agent template (implement this)
├── train_baseline_agent.py        # Reference training script (study this)
├── validate_student_submission.py # Validation script (use this)
└── README.md                      # This instruction file
```

## Quick Start

### 1. Set up the Environment

**Requirements:**
- Python 3.8 or higher
- Required packages: torch, numpy, gymnasium, pettingzoo[atari]

```bash
# Install required packages
pip install torch numpy gymnasium pettingzoo[atari]
```

## Student Requirements

Each student must implement their agent and submit **3 files**:

### Required Files

1. **<entrynumber>_agent.py** - Your agent implementation
   - Must follow the template in student_agent.py
   - Implement the StudentAgent class with required methods
   - Handle RGB observations (210, 160, 3)
   - Output 6 actions (0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=FIRE+RIGHT, 5=FIRE+LEFT)

2. **<entrynumber>_model.pt** - Your trained PyTorch model
   - Saved model weights from your training
   - Must be compatible with your agent architecture

3. **<entrynumber>_train.py** - Your training script
   - Shows how you trained your agent
   - Should demonstrate your training process

### Example Submission Structure

```
2023CS12345_ail821_assignment2/
├── 2023CS12345_agent.py    # Your agent implementation
├── 2023CS12345_model.pt    # Your trained model
└── 2023CS12345_train.py    # Your training script
```

## PettingZoo Pong Environment

### Observation Space
Each agent receives an RGB image observation:
- Shape: (210, 160, 3) - Height, Width, RGB channels
- Type: Raw pixel values from the Atari Pong game
- Range: 0-255 for each color channel

### Action Space
Each agent can take 6 actions (Atari Pong controls):
- 0: NOOP (no operation)
- 1: FIRE (serve ball or spike)
- 2: RIGHT (move paddle right)
- 3: LEFT (move paddle left)
- 4: FIRE+RIGHT (serve and move right)
- 5: FIRE+LEFT (serve and move left)

### Reward Structure
- +1: Score a point (opponent misses ball)
- -1: Concede a point (miss the ball)
- 0: Otherwise

## 🏆 Tournament System

### Round-Robin Format
- Each agent plays against every other agent
- Multiple episodes per match
- 3 points for win, 1 point for draw

### Standard Scoring
- Win: 3 points
- Draw: 1 point
- Loss: 0 points

### Weighted Scoring System
The assignment uses **weighted scoring** for final evaluation:

- **30% weight**: Performance against baseline agent
- **70% weight**: Performance in tournament against other students

**Formula:**
```
Final Score = (0.3 × Baseline_Score) + (0.7 × Tournament_Score)
```

Where:
- **Baseline_Score**: Points earned against the baseline agent (max 3 points per game)
- **Tournament_Score**: Points earned against other student agents (max 3 points per game)

### Why Weighted Scoring?
- **Fair Evaluation**: Rewards both absolute performance and competitive skill
- **Comprehensive Assessment**: Tests agents against known baseline and unknown opponents


## Implementation Details

### Required Agent Interface

You must implement the StudentAgent class in <entrynumber>_agent.py:

```python
class StudentAgent:
    def __init__(self, agent_id, action_space):
        """
        Initialize your multi-agent RL algorithm.

        Args:
            agent_id: ID for this agent (0 or 1)
            action_space: Gym action space (6 discrete actions)
        """
        self.agent_id = agent_id
        self.action_space = action_space
        self.n_actions = 6

        # TODO: Initialize your algorithm (DQN, PPO, MADDPG, etc.)

    def act(self, obs):
        """
        Select an action given RGB observation.

        Args:
            obs: RGB image of shape (210, 160, 3)

        Returns:
            action: Integer (0-5) representing the action
        """
        # TODO: Implement your action selection
        # This function will be used during evaluation
        return 0  # Placeholder

    def update(self, obs, action, reward, next_obs, done):
        """
        Update your agent using a transition.

        Args:
            obs: Current RGB observation
            action: Action taken
            reward: Reward received
            next_obs: Next RGB observation
            done: Whether episode finished
        """
        # TODO: Implement your learning update
        pass

    def save_model(self, filepath):
        """Save your trained model."""
        # TODO: Implement model saving
        pass

    def load_model(self, model_data):
        """Load your trained model."""
        # TODO: Implement model loading
        pass
```

### Neural Network Architecture

You need to process RGB images (210, 160, 3). Consider using:
- CNN layers for visual feature extraction
- Fully connected layers for action values
- Any architecture that works with RGB input

## Testing Your Implementation

### Validate Your Submission

Before submitting, use the validation script to check your implementation:

```bash
python validate_student_submission.py ./your_entry_number_ail821_assignment2
```

This script checks:
- All required files are present
- StudentAgent class is properly implemented
- Model file is valid
- Agent can select actions correctly

### Test with Reference Training Script

Study train_baseline_agent.py to understand:
- How to set up the PettingZoo environment
- How to implement training loops
- How to use the agent interface
- How to save and load models

### Test Your Agent Manually

```python
# Test your agent with the environment
from pettingzoo_pong_wrapper import PettingZooPongWrapper
from student_agent import StudentAgent

env = PettingZooPongWrapper()
obs, _ = env.reset()

# Create your agent
agent = StudentAgent(agent_id=0, action_space=env.action_space[0])

# Test action selection
action = agent.act(obs[0])
print(f"Selected action: {action}")

env.close()
```



## Getting Help

### Resources

1. Study train_baseline_agent.py - Complete training example
2. Study student_agent.py - Template with required interface
3. Study pettingzoo_pong_wrapper.py - Environment usage

### Common Issues

1. RGB Processing
   - Remember to handle (210, 160, 3) RGB images
   - Use CNN layers or preprocessing for visual features
   - Consider normalization or scaling

2. 6-Action Space
   - Actions 0-5 correspond to Atari Pong controls
   - Make sure your agent outputs valid actions (0-5)

3. Training Stability
   - Use appropriate learning rates for CNN training
   - Consider experience replay for stable learning
   - Monitor training progress and loss values

## Expected Performance

A well-implemented agent should:
- Process RGB observations effectively
- Learn meaningful policies from visual input
- Compete in multi-agent scenarios
- Show improvement during training

## Success Criteria

Your implementation is successful if:
- <entrynumber>_agent.py implements required interface
- <entrynumber>_model.pt loads without errors
- <entrynumber>_train.py demonstrates training process
- Agent can select valid actions (0-5)
- Code runs without crashes

## Submission Format

Package your files in a directory named:
```
<entrynumber>_ail821_assignment2/
├── <entrynumber>_agent.py    # Your agent implementation
├── <entrynumber>_model.pt    # Your trained model
└── <entrynumber>_train.py    # Your training script
```

---
