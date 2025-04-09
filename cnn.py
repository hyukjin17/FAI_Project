import torch
import torch.nn as nn
import random
import numpy as np

class MultiPlaneCNN(nn.Module):
    def __init__(self, num_planes, num_actions):
        super(MultiPlaneCNN, self).__init__()
        self.num_planes = num_planes
        self.num_actions = num_actions
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=5, stride=1, padding=2),  # input channels=6
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Compute final flattened size (might want to print this after dummy forward pass)
        self.fc_input_size = 64 * 20 * 20  # Assuming (80x80 grid -> 20x20 after pooling twice)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_planes * num_actions)  # one output per plane-action pair
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = x.view(-1, self.num_planes, self.num_actions)  # Reshape output to (batch_size, num_planes, num_actions)
        return x
    
class MultiPlaneDQNAgent:
    def __init__(self, num_planes, num_actions, gamma=0.99, lr=1e-3):
        self.model = MultiPlaneCNN(num_planes, num_actions)
        self.target_model = MultiPlaneCNN(num_planes, num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.memory = []  # Experience replay buffer

    def store_transition(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def sample_memory(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_memory(batch_size)
        
        states = torch.tensor(np.array(states)).float()
        actions = torch.tensor(np.array(actions)).long()
        rewards = torch.tensor(np.array(rewards)).float()
        next_states = torch.tensor(np.array(next_states)).float()
        dones = torch.tensor(np.array(dones)).float()

        q_values = self.model(states)
        next_q_values = self.target_model(next_states).detach()

        selected_q_values = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        next_max_q_values = next_q_values.max(dim=2)[0]
        target_q_values = rewards + self.gamma * next_max_q_values * (1 - dones)

        loss = self.criterion(selected_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()