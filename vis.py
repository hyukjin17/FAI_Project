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
                if event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                    selected_plane_index = int(event.unicode)  # Plane index from key
                    print(f"Selected plane: {selected_plane_index}")
                elif not game_ended:
                    action = None
                    if event.key == pygame.K_w:
                        action = 2
                    if event.key == pygame.K_s:
                        action = 3
                    if event.key == pygame.K_a:
                        action = 0
                    if event.key == pygame.K_d:
                        action = 1
                    if event.key == pygame.K_u:
                        action = 4
                    if event.key == pygame.K_i:
                        action = 5
                    if event.key == pygame.K_o:
                        action = 6
                    if event.key == pygame.K_p:
                        action = 7
                    if event.key == pygame.K_m:
                        for plane in game.planes:
                            plane.move()
                    if event.key == pygame.K_z:
                        game.add_plane()
                    if action is not None:
                        if selected_plane_index < len(game.planes):
                            actions = [12] * len(game.planes)  # Default all to "go_straight"
                            actions[selected_plane_index] = action
                            game.step(actions)
                        else:
                            print(f"Plane index {selected_plane_index} is out of range.")
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
