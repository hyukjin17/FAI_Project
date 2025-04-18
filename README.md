# AI Air Traffic Control Tower
This project implements 2 different implementations for an agent to manage arriving aircraft at Bradley International Airport. The agent learns to assign runways, control aircraft heading and speed, avoid collisions, and guide planes to the gate using a custom reinforcement learning environment powered by PyTorch and Gymnasium.
The agent is trained using a Deep Q-Network (DQN) approach and a hybrid approach using a Convolutional Neural Network (CNN) and A DQN.

## 🧠 Project Overview

- **Goal**: Safely and efficiently land planes under dynamic wind conditions.
- **Frameworks**: PyTorch, Gymnasium, NumPy, Pygame (for visualization).

- **Agent 1**: Multi-plane DQN agent using a Convolutional Neural Network (CNN).
- **Observation Space 1**: 2D grid of airport with runways, taxiways, planes, and wind simulation.

- **Agent 2**: Multi-plane DQN agent.
- **Observation Space 2**: Per-plane state vector containing the plane, wind, and runway info.

---

## 🧪 How to Run

### 1. Train Agent
- Run the dqn_train.py to train the DQN agent
- Run the cnn_training.py to train the CNN + DQN agent

### 2. Run Trained Agent
- Run the dqn_vis.py to run the trained DQN agent
- Run the test_)trained_agent.py to run the trained CNN + DQN agent
