import torch
import torch.nn as nn
import torch.nn.utils as torch_utils
import random
import numpy as np

class MultiPlaneCNN(nn.Module):
    def __init__(self, num_planes, num_actions):
        super(MultiPlaneCNN, self).__init__()
        self.num_planes = num_planes
        self.num_actions = num_actions
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3, stride=1, padding=1),  # (7, 80, 80) -> (16, 80, 80)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                    # (16, 80, 80) -> (16, 40, 40)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (16, 40, 40) -> (32, 40, 40)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                    # (32, 40, 40) -> (32, 20, 20)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (32, 20, 20) -> (64, 20, 20)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                    # (64, 20, 20) -> (64, 10, 10)
        )

        self.fc_input_size = 64 * 10 * 10  # (after pooling)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_planes * num_actions)  # Output for all planes and actions
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc_layers(x)
        x = x.view(-1, self.num_planes, self.num_actions)  # Reshape output to (batch_size, num_planes, num_actions)
        return x
    
class MultiPlaneDQNAgent:
    def __init__(self, num_planes, num_actions, gamma=0.99, lr=0.0003, max_memory_size=50000):
        self.model = MultiPlaneCNN(num_planes, num_actions)
        self.target_model = MultiPlaneCNN(num_planes, num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.gamma = gamma
        self.memory = []  # Experience replay buffer
        self.max_memory_size = max_memory_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.num_planes = num_planes
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.9)

    def store_transition(self, state, actions, rewards, next_state, done):
        self.memory.append((state, actions, rewards, next_state, done))
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

    def sample_memory(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    # Choose actions for all planes using epsilon-greedy policy.
    def select_actions(self, state, epsilon):
        if random.random() < epsilon:
            # Random actions
            actions = np.random.randint(0, self.model.num_actions, size=self.model.num_planes)
        else:
            # Greedy actions (highest Q-value from network)
            state_tensor = torch.tensor(np.array(state)).float().unsqueeze(0).to(self.device)  # (1, channels, H, W)
            with torch.no_grad():
                q_values = self.model(state_tensor)  # (1, num_planes, num_actions)
            
            q_values = q_values.squeeze(0)  # (num_planes, num_actions)

            actions = q_values.argmax(dim=1)   # compute argmax while still on GPU
            actions = actions.cpu().numpy()    # only THEN move to CPU
        return actions

    def train(self, batch_size=32):
        if len(self.memory) < batch_size * 5:
            return  # skip training for now

        states, actions, rewards, next_states, dones = self.sample_memory(batch_size)
        
        states = torch.tensor(np.array(states)).float().to(self.device)
        actions = torch.tensor(np.array(actions)).long().to(self.device)
        rewards = torch.tensor(np.array(rewards)).float().to(self.device)
        next_states = torch.tensor(np.array(next_states)).float().to(self.device)
        dones = torch.tensor(np.array(dones)).float().to(self.device)

        # Expand dones to (batch_size, num_planes)
        dones = dones.unsqueeze(1).expand(-1, self.num_planes)

        q_values = self.model(states)

        # 1. Select next actions using MAIN model
        next_q_values_main = self.model(next_states)  # (batch_size, num_planes, num_actions)
        next_actions = next_q_values_main.argmax(dim=2, keepdim=True)  # (batch_size, num_planes, 1)

        # 2. Evaluate those actions using TARGET model
        next_q_values_target = self.target_model(next_states)  # (batch_size, num_planes, num_actions)
        next_q_values = next_q_values_target.gather(2, next_actions).squeeze(-1)  # (batch_size, num_planes)

        # 3. Double DQN target
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        target_q_values = target_q_values.detach()  # Safe detach
        target_q_values = torch.clamp(target_q_values, -100, 100) # clamp targets to prevent value explosion
        
        selected_q_values = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        loss = self.criterion(selected_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch_utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
        self.optimizer.step()

        # if hasattr(self, 'scheduler'):
        #     self.scheduler.step()

        self.loss_value = loss.item()

        # Q-value Diagnostic
        current_max_q = q_values.max().item()
        # Create or update a list to store the max Q-values
        if not hasattr(self, 'max_q_list'):
            self.max_q_list = []
        self.max_q_list.append(current_max_q)


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