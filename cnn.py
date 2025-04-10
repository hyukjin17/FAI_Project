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
            nn.Conv2d(7, 32, kernel_size=5, stride=1, padding=2),  # input channels = 7
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Compute final flattened size
        self.fc_input_size = 64 * 20 * 20  # Assuming (80x80 grid -> 20x20 after pooling twice)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_planes * num_actions)  # (batch_size, num_planes, num_actions)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc_layers(x)
        x = x.view(-1, self.num_planes, self.num_actions)  # Reshape output to (batch_size, num_planes, num_actions)
        return x
    
class MultiPlaneDQNAgent:
    def __init__(self, num_planes, num_actions, gamma=0.99, lr=0.001):
        self.model = MultiPlaneCNN(num_planes, num_actions)
        self.target_model = MultiPlaneCNN(num_planes, num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.memory = []  # Experience replay buffer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.num_planes = num_planes

    def store_transition(self, state, actions, rewards, next_state, done):
        self.memory.append((state, actions, rewards, next_state, done))

    def sample_memory(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    # Choose actions for all planes using epsilon-greedy policy.
    def select_actions(self, state, epsilon, num_active_planes=None):
        if random.random() < epsilon:
            # Random actions
            actions = np.random.randint(0, self.model.num_actions, size=self.model.num_planes)
        else:
            # Greedy actions (highest Q-value from network)
            state_tensor = torch.tensor(np.array(state)).float().unsqueeze(0).to(self.device) # (1, channels, H, W)
            with torch.no_grad():
                q_values = self.model(state_tensor)  # (1, num_planes, num_actions)
            q_values = q_values.squeeze(0)  # (num_planes, num_actions)

            actions = q_values.argmax(dim=1)
            actions = actions.squeeze(0).cpu().numpy()
        return actions

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_memory(batch_size)
        
        states = torch.tensor(np.array(states)).float().to(self.device)
        actions = torch.tensor(np.array(actions)).long().to(self.device)
        rewards = torch.tensor(np.array(rewards)).float().to(self.device)
        next_states = torch.tensor(np.array(next_states)).float().to(self.device)
        dones = torch.tensor(np.array(dones)).float().to(self.device)

        # Expand dones to (batch_size, num_planes)
        dones = dones.unsqueeze(1).expand(-1, self.num_planes)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states).detach()

        selected_q_values = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        next_max_q_values = next_q_values.max(dim=2)[0]
        target_q_values = rewards + self.gamma * next_max_q_values * (1 - dones)

        loss = self.criterion(selected_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_value = loss.item()


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Save the model weights to a file
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")

    # Load the model weights from a file
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())
        print(f"Model loaded from {filename}")