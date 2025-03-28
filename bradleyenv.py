import gym
from gym import spaces
import numpy as np
import random
import pygame
from aircraft import Aircraft
import math

class BradleyAirportEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, screen_width=800, screen_height=800):
        super(BradleyAirportEnv, self).__init__()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_aircraft = 10

        # State Space 
        self.x_distance_to_runway = [i for i in range(self.screen_width)]
        self.y_distance_to_runway = [i for i in range(self.screen_height)]
        self.aircraft_size = [0, 1]  # 0: Small, 1: Large
        self.aircraft_speed = [0, 1, 2]  # Speed buckets (low, medium, high)
        self.aircraft_type = [0, 1, 2, 3, 4, 5] # Commercial, cargo, private, military, small
        self.runway_assignment = [0, 1, 2, 3]  # Runway choice and direction (0, 1 for horizontal; 2, 3 for vertical)
        self.wind_speed = [0, 1]  # Low or High
        self.wind_direction = [np.pi/2, np.pi/4, 0, -np.pi/4, -np.pi/2, -3*np.pi/4, np.pi, 3*np.pi/4]  # North, NorthEast, East, SouthEast, South, SouthWest, West, NorthWest
        self.current_state = [0, 1, 2, 3]  # 0: In Air, 1: Taxiway, 2: Runway, 3: At Gate
        self.planes = []
        self.total_planes = 0

        # Observation Space
        self.observation_space = spaces.MultiDiscrete([
            len(self.x_distance_to_runway),
            len(self.y_distance_to_runway),
            len(self.aircraft_size),
            len(self.aircraft_speed),
            len(self.aircraft_type),
            len(self.runway_assignment),
            len(self.wind_speed),
            len(self.wind_direction),
            len(self.current_state)
        ])

        # Action Space
        self.action_space = spaces.Discrete(10)  
        self.actions = {
            0: "turn_left",
            1: "turn_right",
            2: "turn_up",
            3: "turn_down",
            4: "assign_runway_0_direction_0",
            5: "assign_runway_0_direction_1",
            6: "assign_runway_1_direction_0",
            7: "assign_runway_1_direction_1",
            8: "taxi_1",
            9: "taxi_2",
            10: "wait",
            11: "takeoff",
            12: "go_straight"
        }

        self.reset()

    def reset(self):
        self.planes = []
        self.total_planes = 0
        
        self.state = [
            random.choice(self.aircraft_size),
            random.choice(self.aircraft_speed),
            random.choice(self.aircraft_type),
            random.choice(self.runway_assignment),
            random.choice(self.wind_speed),
            random.choice(self.wind_direction),
            0  # Assume initially in air
        ]
        self.time_step = 0
        return np.array(self.state, dtype=np.int32), 0, False
    
    def get_obs(self, plane):
        obs = plane.get_obs()
        self.state = [
            obs[0],
            obs[1],
            obs[2],
            self.runway_assignment,
            self.wind_speed,
            self.wind_direction,
            obs[3]
        ]
        return np.array(self.state, dtype=np.int32)
    
    def move(self, plane, action):
        if action in [0,1,2,3] and plane.flight_state == 2:  # If already near runway
            return 10  # Penalty for unnecessary turns near runway
        crosswind = self.is_within_pi(self.wind_direction, plane.direction)
        if (plane.runway == 0 and 100 < plane.x < 400 and 200 < plane.y < 210) or (
            plane.runway == 1 and 100 < plane.y < 400 and 200 < plane.x < 210):
            plane.move()
            plane.flight_state, self.current_state = 2, 2
            return 100 if crosswind and self.wind_speed == 0 else -100
        
        if action == 0: # turn left
            plane.change_direction(np.pi)
        elif action == 1: # turn right
            plane.change_direction(0)
        elif action == 2: # go up
            plane.change_direction(np.pi/2)
        elif action == 3: # go down
            plane.change_direction(-np.pi/2)
        plane.move()
        return 0
    
    def is_within_pi(theta1, theta2):
        delta_theta = theta1 - theta2
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
        return abs(delta_theta) <= np.pi


    def step(self, plane, action):
        reward = 0
        done = False
        self.time_step += 1
        aircraft_size, aircraft_speed, aircraft_type, runway, wind_speed, wind_dir, current_state = self.get_obs()

        
        if action in [0, 1, 2, 3, 12]:  # Moving
            reward = self.move(plane, action)

        elif action in [4, 5, 6, 7]:  # Assign runway
            reward = self.assign_runway(plane, action)
            if aircraft_size == 1 and action == 4:  # Large aircraft on short runway
                reward -= 100  # Penalty for landing on the wrong runway
            else:
                self.state[2] = action - 4  # Update runway assignment

        elif action in [8, 9]:  # Taxi
            if current_state == 2:  # If already on runway, cannot taxi
                reward -= 10
            else:
                self.state[7] = action - 8  # Move to taxiway

        elif action == 10:  # Wait
            reward -= 1  # Penalty for waiting too long

        # Check for wind direction mismatch
        if (runway_dir != wind_dir):
            reward += 100  # Reward for correct alignment
        else:
            reward -= 100  # Penalty for incorrect wind alignment

        # Check if aircraft is landing at too sharp an angle
        landing_angle = random.randint(-60, 60)  # landing angle
        if current_state == 0 and not (-45 <= landing_angle <= 45):
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
            random.choice(self.current_state)
        ]

        if self.total_planes == self.max_aircraft:
            done = True  

        return np.array(self.state, dtype=np.int32), reward, done
    
    # Add a new plane to the environment
    def add_plane(self):
        plane = Aircraft(self.screen_width, self.screen_height)
        self.planes.append(plane)
        self.total_planes += 1

    def remove_plane(self, plane):
        self.planes.remove(plane)

    def render(self, mode='human'):
        print(
            f"Time Step: {self.time_step} | Wind Speed: {self.wind_speed} | Wind Direction {self.wind_direction} | State: {self.current_state} | Action: {self.actions}")