import gym
from gym import spaces
import numpy as np
import random
from aircraft import Aircraft
from taxiway import Taxiway
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
        self.planes = []
        self.time_step = 0

        # Create two Runway objects
        runway_horizontal = Runway(
            x_start=100, y_start=200,
            x_end=400, y_end=210,
            x_entry = 100, y_entry = 205, # entrance to runway based on direction
            direction=0.0,  # Facing EAST (0 radians)
            name="Runway 0"
        )
        runway_vertical = Runway(
            x_start=200, y_start=100,
            x_end=210, y_end=600,
            x_entry = 205, y_entry = 600, # entrance to runway based on direction
            direction = 3*math.pi / 2,  # Facing NORTH (90 degrees = pi/2 radians)
            name="Runway 1"
        )
        # Create 2 Taxiway objects
        taxiway_horizontal = Taxiway(
            x_start=180, y_start=250,
            x_end=190, y_end=400,
            name="Taxiway 0"
        )
        taxiway_vertical = Taxiway(
            x_start=230, y_start=180,
            x_end=380, y_end=190,
            name="Taxiway 1"
        )

        self.runways = [runway_horizontal, runway_vertical]
        self.taxiways = [taxiway_horizontal, taxiway_vertical]

        self.wind_speeds = [0, 1]  # Low or High
        self.wind_speed = 0
        self.wind_directions = [3*np.pi/2, 7*np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]  # North, NorthEast, East, SouthEast, South, SouthWest, West, NorthWest
        self.wind_direction = random.choice(self.wind_directions)

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
        for _ in range(self.max_aircraft):
            self.add_plane()
        obs = self.generate_state_grid()
        self.time_step = 0
        return obs, 0, False
    

    def move_action(self, plane, action):
        if action == 0: # turn left
            plane.turn("left")
        elif action == 1: # turn right
            plane.turn("right")
        elif action == 2: # speed up
            plane.change_speed(10)
        elif action == 3: # slow down
            plane.change_speed(-10)

        reward = 0
        
        if plane.is_off_screen():
            self.remove_plane(plane)
            reward -= 300   # large penalty for going out of screen

        if plane.runway is None and plane.flight_state == 0:
            return 0
        
        runway = plane.runway
        if runway is not None and abs(plane.direction - runway.direction) < np.pi/4:
            reward += 5  # Small shaping bonus for good early alignment
        
        if plane.flight_state == 0:
            k_penalty = 0.01 # constants for penalty and reward (based on distance to runway)
            k_bonus = 10.0
            reward -= k_penalty * plane.distance_to_runway  # Mild penalty for being far away
            reward += k_bonus / (plane.distance_to_runway + 1)  # Big bonus for getting very close
            if plane.distance_to_runway < 100 and runway.is_occupied():
                reward -= 2 # penalty for moving towards an occupied runway
            if plane.distance_to_runway < 5:
                plane.flight_state = 2
        elif plane.flight_state == 2 and runway is not None: # on runway
            if runway.on_runway(plane.x, plane.y):
                reward += 5 # reward for staying on the runway
            if plane.speed != 0 and plane.runway is not None:
                reward -= plane.speed   # incentivize decelerating to a full stop immediately after landing
            else:
                if runway.name == "Runway 0":
                    taxiway = self.taxiways[0]
                else:
                    taxiway = self.taxiways[1]
                reward -= taxiway.distance_to_center(plane.x, plane.y)  # penalty for being away from taxiway after landing
                if taxiway.close_to(plane.x, plane.y):
                    plane.flight_state = 1
                    plane.stop()
                    reward += 20    # small reward for getting to the taxiway
        elif plane.flight_state == 1:
            reward -= 10 # penalty for assigning a move action on a taxiway

        return reward   # reward roughly between -5 and 5
    

    def is_within_angle(self, theta1, theta2, threshold):
        delta_theta = theta1 - theta2
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi  # Normalize
        return abs(delta_theta) <= threshold
    

    # assigns runway and landing direction
    def assign_runway(self, plane, action):
        """Need a way to change the runway direction if the wind direction is variable"""
        if plane.runway is not None and (
            (action in [4,5] and plane.runway.name == "Runway 0") or (
            action in [6,7] and plane.runway.name == "Runway 1")):
            return 0    # avoid duplicate rewards
        reward = 0
        if action == 4 or action == 5:
            plane.assign_runway(self.runways[0])
            # change direction if not already set correctly based on action
            if (action == 4 and math.isclose(self.runways[0].direction, math.pi, abs_tol=0.001)) or (
                action == 5 and math.isclose(self.runways[0].direction, 0, abs_tol=0.001)):
                self.runways[0].change_direction()
        elif action == 6 or action == 7:
            plane.assign_runway(self.runways[1])
            # change direction if not already set correctly based on action
            if (action == 7 and math.isclose(self.runways[1].direction, 3*math.pi/2, abs_tol=0.001)) or (
                action == 6 and math.isclose(self.runways[1].direction, math.pi/2, abs_tol=0.001)):
                self.runways[1].change_direction()
        if plane.size > 5 and action in [4, 5]:  # Large aircraft on short runway
            reward -= 100   # Penalty for landing on the wrong runway
        else:
            reward += 100   # Reward for assigning the runway correctly
        return reward
    

    # Checks whether there are collisions on the grid and returns the number of collisions
    def check_grid_collisions(self, state_grid):
        plane_layer = state_grid[CHANNELS['plane_presence']]
        collision_cells = plane_layer > 1
        num_collisions = np.sum(collision_cells)
        return num_collisions
        

    # Assign an action to a plane
    def execute_action(self, plane, action):
        reward = 0
        done = False
        self.time_step += 1

        if action in [0, 1, 2, 3, 12]:  # Moving or changing direction
            reward += self.move_action(plane, action)
        elif action in [4, 5, 6, 7]:  # Assign runway
            reward += self.assign_runway(plane, action)
        elif action == 8 and plane.flight_state == 2 and plane.runway is not None:   # Taxi
            if plane.speed != 0:
                reward -= 50    # penalty for assigning a taxiway when the aircraft is not stopped
            if plane.runway.name == "Runway 0":
                taxiway = self.taxiways[0]
            else:
                taxiway = self.taxiways[1]
            plane.runway = None
            if not taxiway.is_occupied():
                x,y = taxiway.get_center()
                direction = math.atan2(y - plane.y, x - plane.x)
                plane.set_direction(direction)
        elif action == 9:   # Go to gate
            if plane.flight_state == 1: # if on taxiway
                reward += 100   # reward for getting to gate
                self.remove_plane(plane)
        elif action == 10:  # Wait
            if plane.flight_state == 0:
                plane.turn("right")
            else:
                plane.stop()
            reward -= 1  # Penalty for waiting too long

        elif action == 11: # Takeoff
            pass
            # currently not used since all planes are starting in air

        plane.move() # Move the plane at each time step
        obs = self.generate_state_grid()

        # Check for collisions
        collisions = self.check_grid_collisions(obs)
        reward -= collisions * 1000
        if collisions > 0 or self.total_planes == 10:
            done = True

        if plane.runway is not None:
            crosswind = self.is_within_angle(self.wind_direction, plane.direction, np.pi/2)
            correct_landing_angle = self.is_within_angle(plane.runway.direction, plane.direction, np.pi/4)
            if (plane.runway.name == "Runway 0" and 100 < plane.x < 400 and 200 < plane.y < 210) or (
                plane.runway.name == "Runway 1" and 100 < plane.y < 600 and 200 < plane.x < 210):
                plane.flight_state = 2
                reward += (-200) if not correct_landing_angle else 100 if crosswind and self.wind_speed == 0 else -100

        all_landed = True
        # checks if all planes have been assigned to gates
        for plane in self.planes:
            if plane.flight_state != 3:
                all_landed = False
        if all_landed:
            done = True
        return obs, reward, done

    def step(self, actions):
        while len(self.planes) < self.max_aircraft:
            self.add_plane()

        obs = self.generate_state_grid()
        per_plane_rewards = []
        done = False

        for plane_index, action in enumerate(actions):
            if plane_index >= len(self.planes):
                continue  # Skip if action is for non-existing plane
            plane = self.planes[plane_index]
            obs, reward, done = self.execute_action(plane, action)
            per_plane_rewards.append(reward)

        # padding values just in case
        while len(per_plane_rewards) < self.max_aircraft:
            per_plane_rewards.append(0)

        # Global reward = sum of all planes
        total_reward = sum(per_plane_rewards)
        return obs, total_reward, per_plane_rewards, done
            
    
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
        if len(self.planes) < self.max_aircraft:
            plane = Aircraft(self.screen_width, self.screen_height)
            self.planes.append(plane)
            self.total_planes += 1

    def remove_plane(self, plane):
        self.planes.remove(plane)

    def render(self, mode='human'):
        print(
            f"Time Step: {self.time_step} | Wind Speed: {self.wind_speed} | Wind Direction {self.wind_direction} | Total Planes: {self.total_planes}")