import gym
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from bradleyenv import BradleyAirportEnv

GAMMA = 0.99
LEARNING_RATE = 0.0001
BUFFER_SIZE = 50000
BATCH_SIZE = 128
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 10000
TARGET_UPDATE = 100
EPISODES = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.policy_net = DQN(self.state_dim, self.action_dim).to(device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START
        self.steps_done = 0

    def select_action(self, state, training=True):
        self.steps_done += 1
        self.epsilon = max(EPSILON_END, EPSILON_START - (self.steps_done / EPSILON_DECAY))

        if random.random() < self.epsilon and training:
            return self.env.action_space.sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                return self.policy_net(state).argmax().item()

    def train_step(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        batch = self.replay_buffer.sample(BATCH_SIZE)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.LongTensor(action).unsqueeze(1).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        done = torch.FloatTensor(done).to(device)

        q_values = self.policy_net(state).gather(1, action).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0].detach()
        target_q_values = reward + (GAMMA * next_q_values * (1 - done))

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def plot_rewards(rewards):
    plt.figure(figsize=(12,6))
    plt.plot(rewards)
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (100-episode)")
    plt.grid()
    plt.show()

def train_dqn():
    env = BradleyAirportEnv()
    agent = DQNAgent(env)
    
    reward_history = []
    avg_reward = deque(maxlen=100)
    
    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            reward = np.clip(reward, -1.0, 1.0)
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            total_reward += reward
        
        avg_reward.append(total_reward)
        reward_history.append(np.mean(avg_reward))
        
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
        
        if episode % 500 == 0 and episode > 0:
            agent.epsilon = max(EPSILON_END, agent.epsilon * 0.9)
        
        if episode % 100 == 0:
            print(f"Ep {episode}: Avg Reward {np.mean(avg_reward):.2f}, Epsilon {agent.epsilon:.3f}")
    
    torch.save(agent.policy_net.state_dict(), "dqn_airport.pth")
    plot_rewards(reward_history)

if __name__ == "__main__":
    train_dqn()
