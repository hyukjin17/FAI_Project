import pygame
import sys
import time
import random
from bradleyenv import BradleyAirportEnv

WIDTH, HEIGHT = 800, 800

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)

# Setup display
screen=None

game_ended = False

fps = 60
sleeptime = 0.1
clock = None

# Initialize simulation
game = BradleyAirportEnv(WIDTH, HEIGHT)

# Initialize Pygame
def setup(GUI=True):
    global screen
    if GUI:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI ATC Simulation")

def main():
    global game_ended
    clock = pygame.time.Clock()
    running = True
    if len(game.planes) == 0:
        game.add_plane()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w and not game_ended:
                    action = 2
                    result = game.step(action)
                if event.key == pygame.K_s and not game_ended:
                    action = 3
                    result = game.step(action)
                if event.key == pygame.K_a and not game_ended:
                    action = 0
                    result = game.step(action)
                if event.key == pygame.K_d and not game_ended:
                    action = 1
                    result = game.step(action)
                if event.key == pygame.K_u and not game_ended:
                    action = 4
                    result = game.step(action)
                if event.key == pygame.K_i and not game_ended:
                    action = 5
                    result = game.step(action)
                if event.key == pygame.K_o and not game_ended:
                    action = 6
                    result = game.step(action)
                if event.key == pygame.K_p and not game_ended:
                    action = 7
                    result = game.step(action)
                if event.key == pygame.K_m and not game_ended:
                    for plane in game.planes:
                        plane.move()
                if event.key == pygame.K_z and not game_ended:
                    game.add_plane()
        screen.fill(BLACK)

        # Draw runways
        pygame.draw.rect(screen, GRAY, (100, 200, 300, 10))  # Horizontal runway
        pygame.draw.rect(screen, GRAY, (200, 100, 10, 300))  # Intersecting vertical runway

        # Draw taxiway
        pygame.draw.rect(screen, DARK_GRAY, (80, 200, 10, 150))
        pygame.draw.rect(screen, DARK_GRAY, (200, 80, 150, 10))

        # Draw each plane
        for plane in game.planes:
            pygame.draw.circle(screen, plane.color, (int(plane.x), int(plane.y)), plane.size)
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()


# Testing the Environment

if __name__ == "__main__":
    setup()
    main()
