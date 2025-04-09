import gym
from gym import spaces
import numpy as np
import random
import pygame
from aircraft import Aircraft
from runway import Runway
import math

GRID_WIDTH = 80
GRID_HEIGHT = 80
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

NUM_CHANNELS = 7

CHANNELS = {
    'plane_presence': 0,
    'plane_sin_heading': 1,
    'plane_cos_heading': 2,
    'plane_speed': 3,
    'plane_size': 4,
    'runway_presence': 5,
    'runway_direction': 6,
}

class BradleyAirportEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT):
        super(BradleyAirportEnv, self).__init__()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_aircraft = 5
        self.total_planes = 0

        # Create two Runway objects
        runway_horizontal = Runway(
            x_start=100, y_start=200,
            x_end=400, y_end=210,  # x_start + width, y_start + height
            x_entry = 100, y_entry = 205, # entrance to runway based on direction
            direction=0.0,  # Facing EAST (0 radians)
            name="Runway 0"
        )

        runway_vertical = Runway(
            x_start=200, y_start=100,
            x_end=210, y_end=600,  # x_start + width, y_start + height
            x_entry = 205, y_entry = 600, # entrance to runway based on direction
            direction = math.pi / 2,  # Facing NORTH (90 degrees = π/2 radians)
            name="Runway 1"
        )

        self.runways = [runway_horizontal, runway_vertical]

        # # State Space 
        # self.x_distance_to_runway = [i for i in range(self.screen_width)]
        # self.y_distance_to_runway = [i for i in range(self.screen_height)]
        # self.aircraft_size = [0, 1]  # 0: Small, 1: Large
        # self.aircraft_speed = [0, 1, 2]  # Speed buckets (low, medium, high)
        # self.aircraft_type = [0, 1, 2, 3, 4, 5] # Commercial, cargo, private, military, small
        # # 0 for left, 1 for right, 2 for bottom, 3 for top entry of the runways
        # self.runway_assignment = [0, 1, 2, 3]  # Runway choice and direction (0, 1 for horizontal; 2, 3 for vertical)
        self.wind_speed = [0, 1]  # Low or High
        self.wind_direction = [np.pi/2, np.pi/4, 0, 7*np.pi/4, 3*np.pi/2, 5*np.pi/4, np.pi, 3*np.pi/4]  # North, NorthEast, East, SouthEast, South, SouthWest, West, NorthWest
        # self.current_state = [0, 1, 2, 3]  # 0: In Air, 1: Taxiway, 2: Runway, 3: At Gate
        # self.planes = []
        # self.total_planes = 0

        # # Observation Space
        # self.observation_space = spaces.MultiDiscrete([
        #     len(self.x_distance_to_runway),
        #     len(self.y_distance_to_runway),
        #     len(self.aircraft_size),
        #     len(self.aircraft_speed),
        #     len(self.aircraft_type),
        #     len(self.runway_assignment),
        #     len(self.wind_speed),
        #     len(self.wind_direction),
        #     len(self.current_state)
        # ])

        # Action Space
        num_actions = 13
        self.action_space = spaces.MultiDiscrete([num_actions for _ in range(self.max_aircraft)])
        self.actions = {
            0: "turn_left",
            1: "turn_right",
            2: "speed_up",
            3: "slow_down",
            4: "assign_runway_0_direction_0",
            5: "assign_runway_0_direction_1",
            6: "assign_runway_1_direction_0",
            7: "assign_runway_1_direction_1",
            8: "taxi",
            9: "go_to_gate",
            10: "wait",
            11: "takeoff",
            12: "go_straight"
        }

        self.reset()
        self.add_plane()

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
        crosswind = self.is_within_pi(self.wind_direction, plane.direction)
        if plane.runway == 0:
            correct_landing_angle = True if 7*np.pi/4 < plane.direction < 2*np.pi or 0 < plane.direction < np.pi/4 else False
        elif plane.runway == 1:
            correct_landing_angle = True if 3*np.pi/4 < plane.direction < 5*np.pi/4 else False
        elif plane.runway == 2:
            correct_landing_angle = True if np.pi/4 < plane.direction < 3*np.pi/4 else False
        elif plane.runway == 3:
            correct_landing_angle = True if 5*np.pi/4 < plane.direction < 7*np.pi/4 else False
        if (plane.runway in [0,1] and 100 < plane.x < 400 and 200 < plane.y < 210) or (
            plane.runway in [2,3] and 100 < plane.y < 400 and 200 < plane.x < 210):
            plane.flight_state, self.current_state = 2, 2
            return -200 if not correct_landing_angle else 100 if crosswind and self.wind_speed == 0 else -100
        
        if action == 0: # turn left
            plane.turn("left")
        elif action == 1: # turn right
            plane.turn("right")
        elif action == 2: # speed up
            plane.change_speed(10)
            plane.move()
        elif action == 3: # slow down
            plane.change_speed(-10)
            plane.move()
        elif action == 12:
            plane.move()
        return 0
    
    def is_within_pi(theta1, theta2):
        delta_theta = theta1 - theta2
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
        return abs(delta_theta) <= np.pi
    
    # assigns runway and landing direction
    def assign_runway(self, plane, action):
        if action == 4 or action == 5:
            plane.assign_runway(self.runways[0])
            # change direction if not already set correctly based on action
            if (action == 4 and math.isclose(self.runways[0].direction, math.pi, 0.001)) or (
                action == 5 and math.isclose(self.runways[0].direction, 0, 0.001)):
                self.runways[0].change_direction()
        elif action == 6 or action == 7:
            plane.assign_runway(self.runways[1])
            # change direction if not already set correctly based on action
            if (action == 6 and math.isclose(self.runways[1].direction, 3*math.pi/2, 0.001)) or (
                action == 7 and math.isclose(self.runways[1].direction, math.pi/2, 0.001)):
                self.runways[1].change_direction()

    def execute_action(self, plane, action):
        reward = 0
        done = False
        self.time_step += 1
        aircraft_size, aircraft_speed, aircraft_type, runway, wind_speed, wind_dir, current_state = self.get_obs(plane)
        
        if action in [0, 1, 2, 3, 12]:  # Moving or changing direction
            reward = self.move(plane, action)

        elif action in [4, 5, 6, 7]:  # Assign runway
            if plane.size > 5 and action in [4, 5]:  # Large aircraft on short runway
                reward -= 100  # Penalty for landing on the wrong runway
            else:
                # Update runway assignment
                reward = self.assign_runway(plane, action)

        elif action in [8, 9]:  # Taxi
            self.state[7] = action - 8  # Move to taxiway
            plane.runway = None
            # Need to move to taxiway and set new state

        elif action == 10:  # Wait
            if plane.flight_state == 0:
                plane.turn("right")
            else:
                plane.stop()
            reward -= 1  # Penalty for waiting too long

        elif action == 11: # Takeoff
            pass
            # need to make sure state is updated and plane starts to move

        plane.move() # Move the plane at each time step
        obs = self.generate_state_grid()

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

        all_landed = True
        # checks if all planes have been assigned to gates
        for plane in self.planes:
            if plane.flight_state != 3:
                all_landed = False
        if all_landed:
            done = True
        return obs, reward, done

    def step(self, actions):
        for plane_index, action in enumerate(actions):
            plane = self.planes[plane_index]
            self.execute_action(plane, action)
            if plane.is_off_screen():
                self.remove_plane(plane)
    
    def generate_state_grid(self):
        state = np.zeros((NUM_CHANNELS, GRID_WIDTH, GRID_HEIGHT))

        # Populate plane data
        for plane in self.planes:
            grid_x = int(plane.x / (SCREEN_WIDTH / GRID_WIDTH))
            grid_y = int(plane.y / (SCREEN_HEIGHT / GRID_HEIGHT))

            if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                state[CHANNELS['plane_presence']][grid_x][grid_y] = 1
                state[CHANNELS['plane_sin_heading']][grid_x][grid_y] = math.sin(plane.direction) # sin encoding plane direction
                state[CHANNELS['plane_cos_heading']][grid_x][grid_y] = math.cos(plane.direction) # cos encoding plane direction
                state[CHANNELS['plane_speed']][grid_x][grid_y] = plane.speed
                state[CHANNELS['plane_size']][grid_x][grid_y] = plane.size

        # Populate runway data
        for runway in self.runways:
            x_start = int(runway.x_start / (SCREEN_WIDTH / GRID_WIDTH))
            x_end = int(runway.x_end / (SCREEN_WIDTH / GRID_WIDTH))
            y_start = int(runway.y_start / (SCREEN_HEIGHT / GRID_HEIGHT))
            y_end = int(runway.y_end / (SCREEN_HEIGHT / GRID_HEIGHT))

            for x in range(x_start, min(x_end + 1, GRID_WIDTH)):
                for y in range(y_start, min(y_end + 1, GRID_HEIGHT)):
                    state[CHANNELS['runway_presence']][x][y] = 1
                    state[CHANNELS['runway_direction']][x][y] = math.sin(runway.direction)  # sin encoding runway direction
        return state

    # Add a new plane to the environment
    def add_plane(self):
        if self.planes.size < self.max_aircraft:
            plane = Aircraft(self.screen_width, self.screen_height)
            self.planes.append(plane)
            self.total_planes += 1

    def remove_plane(self, plane):
        self.planes.remove(plane)

    def render(self, mode='human'):
        print(
            f"Time Step: {self.time_step} | Wind Speed: {self.wind_speed} | Wind Direction {self.wind_direction} | Total Planes: {self.total_planes}")