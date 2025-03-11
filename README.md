# Microswimmer Reinforcement Learning

A reinforcement learning framework for controlling microswimmers in 2D turbulent flow environments. This project explores how intelligent agents can navigate complex fluid dynamics to reach target destinations more efficiently than traditional control methods.

![Microswimmer Navigation](https://github.com/yourusername/rl-swimmer/raw/main/images/microswimmer-demo.gif)

## Overview

Microswimmers are small-scale agents that move through fluid environments. Controlling their navigation in turbulent flows presents significant challenges due to the complex and often unpredictable nature of fluid dynamics. This project implements:

1. Custom OpenAI Gym environments for simulating microswimmers in 2D turbulent flows
2. Reinforcement learning algorithms to train intelligent control policies
3. Comparison with naive (direct) control strategies
4. Visualization tools for analyzing agent behavior and flow patterns

## Features

- **Custom Gym Environments**: `MicroSwimmer-v0` and `MicroSwimmer-v1` for different simulation complexities
- **Multiple Control Strategies**:
  - Reinforcement Learning (RL) agent with DQN
  - Naive/direct controller (always points toward target)
- **Physics-Based Simulation**:
  - Realistic fluid dynamics with vorticity fields
  - Bilinear interpolation for smoother transitions
  - Configurable swimmer parameters (speed, sensitivity)
- **Visualization Tools**:
  - Real-time rendering of swimmers and flow fields
  - Animation capabilities to record agent trajectories
  - Comparative performance analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rl-swimmer.git
cd rl-swimmer

# Install dependencies
pip install -r requirements.txt

# Install the local package
pip install -e .
```

## Requirements

- Python 3.6+
- TensorFlow 2.x
- Gym
- Matplotlib
- NumPy
- SciPy
- Keras-RL

## Usage

### Training a DQN Agent

```python
import gym
import envs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Create environment
env = gym.make('MicroSwimmer-v1')

# Build model
model = Sequential([
    Flatten(input_shape=(1, 9)),
    Dense(14, activation='relu'),
    Dense(8, activation='relu')
])

# Create agent
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(
    model=model, 
    memory=memory, 
    policy=policy,
    nb_actions=8, 
    target_model_update=1e-3
)
dqn.compile(optimizer='adam', metrics=['mse'])

# Train
dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)

# Save weights
dqn.save_weights('dqn_weights.h5f', overwrite=True)
```

### Testing an Agent

```python
# Load weights
dqn.load_weights('dqn_weights.h5f')

# Test
scores = dqn.test(env, nb_episodes=10, visualize=True)
```

### Visualizing Results

```python
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from envs.utils import animate_gym

fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
anim = anm.FuncAnimation(
    fig, 
    lambda j: animate_gym(
        j, 
        env.xx, 
        env.yy, 
        env.omg_precompute, 
        env.rl_traj, 
        env.rl_ctrl, 
        env.naive_traj, 
        env.naive_ctrl, 
        env.target
    ), 
    frames=env.t, 
    interval=1, 
    repeat=False
)
plt.show()

# Save animation
writergif = anm.PillowWriter(fps=30)
anim.save('simulation.gif', writer=writergif)
```

## Environment Details

### State Space

The microswimmer state includes:
- Position (x, y)
- Direction vector
- Local vorticity
- Target position
- Time step

### Action Space

The agent can choose from 8 swimming directions:
- Direct toward target (0°)
- 45° rotation from target direction
- 90° rotation from target direction
- 135° rotation from target direction
- 180° rotation from target direction (away from target)
- 225° rotation from target direction
- 270° rotation from target direction
- 315° rotation from target direction

### Reward Function

The primary reward compares the RL agent's performance to a naive controller, incentivizing the agent to find more efficient paths to the target through the turbulent flow.

## Project Structure

```
rl-swimmer/
├── envs/
│   ├── __init__.py     # Environment registration
│   ├── swimmer.py      # Microswimmer environments
│   └── utils.py        # Helper functions
├── main_deeprl.py      # DQN training script
├── test.py             # Testing script
├── precompute4000_equal_vortices.mat  # Precomputed flow fields
└── README.md           # This file
```



## Acknowledgments

- The reinforcement learning framework builds on Keras-RL and OpenAI Gym
