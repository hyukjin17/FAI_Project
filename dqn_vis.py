import pygame
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bradleyenv import BradleyAirportEnv
from collections import Counter
import imageio  


action_counter = Counter()
WIDTH, HEIGHT = 800, 800
MAX_PLANES = 5
STATE_DIM = 6 * MAX_PLANES
ACTION_DIM = 13 * MAX_PLANES
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
OLIVE = (162, 148, 119)

class DQN(nn.Module):
    def __init__(self, input_dim=STATE_DIM, output_dim=ACTION_DIM):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

env = BradleyAirportEnv(WIDTH, HEIGHT)
model = DQN(input_dim=STATE_DIM, output_dim=ACTION_DIM).to(device)
model.load_state_dict(torch.load("centralized_dqn_airport.pth", map_location=device))
model.eval()


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Centralized DQN Tower Controller")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

if len(env.planes) == 0:
    env.add_plane()
for _ in range(3):  
    env.add_plane()

frames = []

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)

    
    pygame.draw.rect(screen, GRAY, (100, 200, 300, 10))
    pygame.draw.rect(screen, GRAY, (200, 100, 10, 300))
    pygame.draw.rect(screen, DARK_GRAY, (80, 200, 10, 150))
    pygame.draw.rect(screen, DARK_GRAY, (200, 80, 150, 10))
    
    for gx, gy in env.gate_zones.values():
        pygame.draw.rect(screen, CYAN, pygame.Rect(gx - 10, gy - 10, 20, 20))
    
    for entry in env.runway_entries.values():
        pygame.draw.circle(screen, BLUE, entry, 6)
    
    for exit in env.runway_exits.values():
        pygame.draw.circle(screen, OLIVE, exit, 6)

    
    obs = []
    for plane in env.planes[:5]:  
        obs.extend(env.get_obs(plane))

    if len(obs) < STATE_DIM:
        obs += [0.0] * (STATE_DIM - len(obs))
    else:
        obs = obs[:STATE_DIM]

    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(obs_tensor).squeeze(0)
    actions = []
    for i in range(len(env.planes)):
        action_slice = q_values[i * 13: (i + 1) * 13]
        action = torch.argmax(action_slice).item()
        actions.append(action)
    
    for action in actions:
        action_counter[action] += 1

    if pygame.time.get_ticks() % 3000 < 30:  
        print("Action counts so far:", dict(action_counter))

    env.step(actions)

    for i, plane in enumerate(env.planes):
        pygame.draw.circle(screen, plane.color, (int(plane.x), int(plane.y)), plane.size)
        end_x = int(plane.x + 15 * plane.dx)
        end_y = int(plane.y + 15 * plane.dy)
        pygame.draw.line(screen, RED, (int(plane.x), int(plane.y)), (end_x, end_y), 2)

        speed_text = font.render(f"Speed: {int(plane.speed)}", True, WHITE)
        screen.blit(speed_text, (int(plane.x + 10), int(plane.y + 10)))
        if hasattr(plane, "gate_target"):
            gx, gy = plane.gate_target
            pygame.draw.circle(screen, YELLOW, (int(gx), int(gy)), 5)  
            gate_label = font.render(f"G{i}", True, WHITE)
            screen.blit(gate_label, (int(plane.x + 10), int(plane.y + 10)))

    frame = pygame.surfarray.array3d(pygame.display.get_surface())
    frame = np.transpose(frame, (1, 0, 2))  
    frames.append(frame)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
print("Saving gif...")
imageio.mimsave('airport_simulation.gif', frames, fps=30)  
print("GIF saved as 'airport_simulation.gif'!")


