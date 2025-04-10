import numpy as np
import torch
import pygame
import time
import random
from bradleyenv import BradleyAirportEnv
from aircraft import Aircraft
from cnn import MultiPlaneDQNAgent

# Hyperparameters
num_planes = 5
num_actions = 13
model_path = "dqn_airport_final.pth"

# Pygame setup
WIDTH, HEIGHT = 800, 800
fps = 30
screen = None

GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)

def setup_pygame():
    global screen
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Trained DQN Airport Simulation")

def render_env(env):
    screen.fill((0, 0, 0))  # Black background

    # Draw runways
    for runway in env.runways:
        pygame.draw.rect(
            screen, GRAY, (runway.x_start, runway.y_start, runway.x_end - runway.x_start, runway.y_end - runway.y_start))

    # Draw taxiways
    for taxiway in env.taxiways:
        pygame.draw.rect(
            screen, DARK_GRAY, (taxiway.x_start, taxiway.y_start, taxiway.x_end - taxiway.x_start, taxiway.y_end - taxiway.y_start))
        

    # Draw planes
    for plane in env.planes:
        pygame.draw.circle(screen, plane.color, (int(plane.x), int(plane.y)), plane.size)

    pygame.display.flip()

def test_trained_agent():
    # Initialize environment and agent
    env = BradleyAirportEnv()
    agent = MultiPlaneDQNAgent(num_planes, num_actions)
    agent.load(model_path)

    # Reset environment
    state = env.reset()
    state = env.generate_state_grid()

    done = False
    max_steps = 500
    step_count = 0

    setup_pygame()
    clock = pygame.time.Clock()

    while not done and step_count < max_steps:
        # Add new planes at random
        if random.random() < 0.05:
            plane = Aircraft(WIDTH, HEIGHT)
            env.add_plane(plane)

        # Select actions (purely greedy, no randomness)
        actions = agent.select_actions(state, epsilon=0.0)

        # Step environment
        next_state, reward, done, info = env.step(actions)
        next_state = env.generate_state_grid()

        # Render the environment
        render_env(env)
        clock.tick(fps)  # Maintain FPS

        # Update for next loop
        state = next_state
        step_count += 1

        # Allow quitting manually
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    pygame.quit()
    print("Testing finished!")

if __name__ == "__main__":
    test_trained_agent()