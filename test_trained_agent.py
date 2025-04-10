import numpy as np
import pygame
import random
import imageio
from bradleyenv import BradleyAirportEnv
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
    state, _, _ = env.reset()
    state = env.generate_state_grid()

    done = False
    max_steps = 10000
    step_count = 0

    setup_pygame()
    clock = pygame.time.Clock()
    frames = []

    while not done and step_count < max_steps:
        # Always maintain 5 planes
        while len(env.planes) < env.max_aircraft:
            env.add_plane()

        # Select actions (purely greedy, no randomness)
        actions = agent.select_actions(state, epsilon=0.0)

        # Step environment
        next_state, total_reward, per_plane_rewards, done = env.step(actions)

        # Render the environment
        render_env(env)

        # Save the current frame
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.transpose(frame, (1, 0, 2))  # Pygame has (width, height), need (height, width)
        frames.append(frame)

        clock.tick(fps)  # Maintain FPS

        # Update for next loop
        state = next_state
        step_count += 1

        # Allow quitting manually
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    pygame.quit()
    # Save all frames into a gif
    print("Saving gif...")
    imageio.mimsave('airport_simulation.gif', frames, fps=fps)
    print("GIF saved as 'airport_simulation.gif'!")

if __name__ == "__main__":
    test_trained_agent()