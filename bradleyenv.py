import gym
from gym import spaces
import numpy as np
import random
import pygame
from aircraft import Aircraft

class BradleyAirportEnv(gym.Env):
    def __init__(self, screen_width=800, screen_height=800):
        super(BradleyAirportEnv, self).__init__()
        self.screen_width = screen_width
        self.screen_height = screen_height

        # State Space 
        self.aircraft_size = [0, 1]  # 0: Small, 1: Large
        self.aircraft_speed = [0, 1, 2]  # Speed buckets (low, medium, high)
        self.aircraft_type = [0, 1, 2, 3, 4, 5] # Commercial, cargo, private, military, small
        self.runway_assignment = [0, 1]  # Runway choice (0 or 1)
        self.runway_direction = [0, 1]  # Runway facing direction (+ or - coordinate)
        self.wind_speed = [0, 1]  # Low or High
        self.wind_direction = [0, 1, 2, 3, 4, 5, 6, 7]  # North, NorthEast, East, SouthEast, South, SouthWest, West, NorthWest
        self.takeoff_or_landing = [0, 1, 2]  # 0: Takeoff, 1: Landing, 2: Neither
        self.current_state = [0, 1, 2, 3]  # 0: In Air, 1: Taxiway, 2: Runway, 3: At Gate
        self.planes = []

        # Observation Space
        self.observation_space = spaces.MultiDiscrete([
            len(self.aircraft_size),
            len(self.aircraft_speed),
            len(self.aircraft_type),
            len(self.runway_assignment),
            len(self.runway_direction),
            len(self.wind_speed),
            len(self.wind_direction),
            len(self.takeoff_or_landing),
            len(self.current_state)
        ])

        # Action Space
        self.action_space = spaces.Discrete(10)  
        self.actions = {
            0: "turn_left",
            1: "turn_right",
            2: "turn_up",
            3: "turn_down",
            4: "assign_runway_0",
            5: "assign_runway_1",
            6: "assign_runway_direction_0",
            7: "assign_runway_direction_1",
            8: "taxi",
            9: "wait"
        }

        self.reset()

    def reset(self):
        self.planes = []
        
        self.state = [
            random.choice(self.aircraft_size),
            random.choice(self.aircraft_speed),
            random.choice(self.aircraft_type),
            random.choice(self.runway_assignment),
            random.choice(self.runway_direction),
            random.choice(self.wind_speed),
            random.choice(self.wind_direction),
            random.choice(self.takeoff_or_landing),
            0  # Assume initially in air
        ]
        self.time_step = 0
        return np.array(self.state, dtype=np.int32), 0, False
    
    def get_obs(self):
        self.state = [
            self.aircraft_size,
            self.aircraft_speed,
            self.aircraft_type,
            self.runway_assignment,
            self.runway_direction,
            self.wind_speed,
            self.wind_direction,
            self.takeoff_or_landing,
            self.current_state
        ]
        return np.array(self.state, dtype=np.int32)

    def step(self, action):
        
        reward = 0
        done = False
        self.time_step += 1
        aircraft_size, aircraft_speed, runway, runway_dir, wind_speed, wind_dir, mode, current_state = self.state

        
        if action in [0, 1, 2, 3]:  # Turning
            if self.time_step > 1 and current_state == 2:  # If already near runway
                reward -= 10  # Penalty for unnecessary turns near runway

        elif action in [4, 5]:  # Assign runway
            if aircraft_size == 1 and action == 4:  # Large aircraft on short runway
                reward -= 100  # Penalty for landing on the wrong runway
            else:
                self.state[2] = action - 4  # Update runway assignment

        elif action in [6, 7]:  # Assign runway direction
            self.state[3] = action - 6  # Update runway direction

        elif action == 8:  # Taxi
            if current_state == 2:  # If already on runway, cannot taxi
                reward -= 10
            else:
                self.state[7] = 1  # Move to taxiway

        elif action == 9:  # Wait
            reward -= 10  # Penalty for waiting too long

        # Check for wind direction mismatch
        if (runway_dir != wind_dir):
            reward += 100  # Reward for correct alignment
        else:
            reward -= 100  # Penalty for incorrect wind alignment

        # Check if aircraft is landing at too sharp an angle
        landing_angle = random.randint(-60, 60)  # landing angle
        if mode == 1 and not (-45 <= landing_angle <= 45):
            reward -= 200  # Penalty for sharp landing

        # Check for collisions
        if random.random() < 0.05:  # Simulated 5% chance of aircraft collision
            reward -= 1000  # Major penalty for crashes
            done = True

        # Update state randomly for simulation purposes
        self.state = [
            aircraft_size,
            random.choice(self.aircraft_speed),
            self.state[2],  # Keep runway
            self.state[3],  # Keep direction
            random.choice(self.wind_speed),
            random.choice(self.wind_direction),
            mode,
            random.choice(self.current_state)
        ]

        if self.time_step >= 50:
            done = True  

        return np.array(self.state, dtype=np.int32), reward, done
    
    # Add a new plane to the environment
    def add_plane(self):
        if self.traffic < self.max_aircraft:
            plane = Aircraft(self.screen_width, self.screen_height)
            self.planes.append(plane)

    def remove_plane(self, plane):
        self.planes.remove(plane)

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((500, 500))
                self.clock = pygame.time.Clock()

            self.screen.fill((0, 0, 0))  # Clear screen
            env.update() # update the positions to next time tick

            # Draw runways
            pygame.draw.rect(self.screen, (200, 200, 200), (100, 200, 300, 10))  # Horizontal runway
            pygame.draw.rect(self.screen, (200, 200, 200), (200, 100, 10, 300))  # Intersecting vertical runway

            # Draw taxiway
            pygame.draw.rect(self.screen, (100, 100, 100), (80, 200, 10, 150))  # Parallel taxiway

            # Draw each plane
            for plane in self.planes:
                pygame.draw.circle(self.screen, plane.color, (int(plane.x), int(plane.y)), plane.size)

        print(
            f"Time Step: {self.time_step} | Wind Speed: {self.wind_speed} | Wind Direction {self.wind_direction} | State: {self.current_state} | Action: {self.actions}")


# Testing the Environment

if __name__ == "__main__":
    env = BradleyAirportEnv()
    
    num_episodes = 5
    max_steps = 20  

    for episode in range(num_episodes):
        obs, reward, done = env.reset()
        print(f"\n==== Episode {episode + 1} ====")

        for step in range(max_steps):
            action = env.action_space.sample()  
            next_obs, reward, done, _ = env.step(action)

            print(f"Step {step + 1}: Action={env.actions[action]}, Reward={reward}, Next State={next_obs}")
            env.render()

            if done:
                print(f"Episode {episode + 1} finished after {step + 1} steps.")
                break

    env.close()
