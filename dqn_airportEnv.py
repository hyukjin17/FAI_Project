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
        self.max_aircraft = 2

        self.aircraft_size = [0, 1]  # 0: Small, 1: Large
        self.aircraft_speed = [0, 1, 2]
        self.aircraft_type = [0, 1, 2, 3, 4]
        self.runway_assignment = [0, 1, 2, 3]
        self.wind_speed_options = [0, 1]
        self.wind_direction_options = [
            np.pi/2, np.pi/4, 0, -np.pi/4,
            -np.pi/2, -3*np.pi/4, np.pi, 3*np.pi/4
        ]
        self.current_state_options = [0, 1, 2, 3]  # In Air, Taxiway, Runway, At Gate

        self.vertical_taxiway = (80, 200, 10, 150)
        self.horizontal_taxiway = (200, 80, 150, 10)
        self.vertical_taxi_center = (85, 275)
        self.horizontal_taxi_center = (275, 85)

        # Gate positions at four corners
        self.gate_zones = {
            0: (self.screen_width - 80, self.screen_height - 80),  # gate for top entry
            1: (self.screen_width - 120, self.screen_height - 120) # gate for left entry
        }

        self.runway_entries = {
            4: (205, 100),  # Vertical (Top Entry)
            5: (205, 100),  
            6: (100, 205),  # Horizontal (Left Entry)
            7: (100, 205),  
        }

        self.runway_exits = {
            4: (205, 400),  # Vertical (Bottom Exit)
            5: (205, 400),
            6: (400, 205),  # Horizontal (Right Exit)
            7: (400, 205),
        }

        self.planes = []
        self.total_planes = 0
        self.time_step = 0
        self.runway_occupied = False  # global flag

        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Discrete(13)

        self.actions = {
            0: "turn_left", 1: "turn_right", 2: "speed_up", 3: "slow_down",
            4: "assign_runway_0_direction_0", 5: "assign_runway_0_direction_1",
            6: "assign_runway_1_direction_0", 7: "assign_runway_1_direction_1",
            8: "taxi", 9: "go_to_gate", 10: "wait", 11: "takeoff", 12: "go_straight"
        }

        self.reset()

    def reset(self):
        self.planes = []
        self.total_planes = 0
        self.time_step = 0
        self.runway_occupied = False

        self.wind_speed = random.choice(self.wind_speed_options)
        self.wind_direction = random.choice(self.wind_direction_options)
        self.state = [0, 0, 0, 0, self.wind_speed, self.wind_direction, 0]

        self.add_plane()
        return np.array(self.state, dtype=np.float32), 0, False, {}

    def get_obs(self, plane):
        obs = plane.get_obs()  # [size_class, speed_bucket, type_idx, flight_state]
        # Compute relative position to runway center
        runway_center_x, runway_center_y = 250, 205
        dx = (runway_center_x - plane.x) / self.screen_width
        dy = (runway_center_y - plane.y) / self.screen_height

        taxi_x, taxi_y = self.vertical_taxi_center
        dx_taxi = (taxi_x - plane.x) / self.screen_width
        dy_taxi = (taxi_y - plane.y) / self.screen_height

        return np.array([
            plane.x / self.screen_width,
            plane.y / self.screen_height,
            obs[0],  # size
            obs[1],  # speed bucket
            obs[3],  # flight state
            plane.speed / 400,  # normalized
            1.0 if self.runway_occupied else 0.0,
            dx,
            dy,
            dx_taxi,
            dy_taxi
        ], dtype=np.float32)

    def move(self, plane: Aircraft, action):
        reward = 0

        if plane.flight_state in [1, 2]:
            reward -= 1  # discourage unnecessary movement

        pre_align = plane.align  # store previous alignment of the plane with the wind

        if action == 0:
            plane.turn("left")
        elif action == 1:
            plane.turn("right")
        elif action == 2:
            plane.change_speed(10)
            reward += 0.5  
        elif action == 3:
            plane.change_speed(-10)
            if plane.speed > 100 and plane.flight_state == 0:
                reward += 1 
            elif plane.speed < 50:
                reward -= 0.5  
        elif action == 12:
            post_align = self.is_within_pi(self.wind_direction, plane.direction)
            too_close = any(
                other != plane and plane.distance_to(other.x, other.y) < 50
                for other in self.planes
            )

            if plane.flight_state == 1 and self.is_on_taxiway(plane):
                if not too_close:
                    reward += 0.5  # Safe movement on taxiway
                else:
                    reward -= 0.2  # taxiway traffic jam

            elif pre_align and not too_close: # Aligned and safe
                reward += 0.5  
            elif pre_align and too_close: # Going straight but close to another plane
                reward -= 0.5  
            else:
                reward -= 0.2 

        post_align = self.is_within_pi(self.wind_direction, plane.direction)
        if action in [0, 1] and not pre_align and post_align: # reward for turn actions
            reward += 2

        # Update 
        plane.align = post_align

        return reward

    @staticmethod
    def is_within_pi(theta1, theta2):
        delta = (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi
        return abs(delta) <= np.pi / 4

    def is_on_runway(self, plane):
        # For vertical runway (top → bottom)
        if plane.runway in [4, 5]:  
            return 200 < plane.x < 210 and 100 < plane.y < 400
        # For horizontal runway (left → right)
        elif plane.runway in [6, 7]:  
            return 100 < plane.x < 400 and 200 < plane.y < 210
        return False
        
    def gate_assign(self, action):
        if action in [4, 5]:  # Vertical 
            return self.gate_zones[0]
        elif action in [6, 7]:  # Horizontal 
            return self.gate_zones[1]
        return self.gate_zones[0]
      
    def is_at_gate(self, plane):
        if not hasattr(plane, "gate_target"):
            return False
        gx, gy = plane.gate_target
        return abs(plane.x - gx) < 20 and abs(plane.y - gy) < 20
    
    # Function to find the nearest entry based on the pos of the plane, return runway_id and angle
    def find_entry(self, plane):
        options = [
            (4, np.pi / 2),  # Vertical
            (6, 0),          # Horizontal
        ]

        best_score = -float('inf')
        best_option = (4, np.pi / 2)

        for runway_id, angle in options:
            aligned = self.is_within_pi(plane.direction, angle)
            dx = plane.x - 250
            dy = plane.y - 205
            dist = math.sqrt(dx**2 + dy**2)
            score = (2 if aligned else 0) - 0.01 * dist
            if score > best_score:
                best_score = score
                best_option = (runway_id, angle)

        return best_option
    
    def check_collisions(self, min_dist=30):
        for i, p1 in enumerate(self.planes):
            for j, p2 in enumerate(self.planes):
                if i < j and p1.distance_to(p2.x, p2.y) < min_dist:
                    return True
        return False
    
    def is_on_taxiway(self, plane):
        vertical_taxi = (80 <= plane.x <= 90 and 200 <= plane.y <= 350)
        horizontal_taxi = (200 <= plane.x <= 350 and 80 <= plane.y <= 90)
        return vertical_taxi or horizontal_taxi

    def execute_action(self, plane: Aircraft, action):
        reward = 0
        done = False
        self.time_step += 1
        obs = self.get_obs(plane)

        # Reward for plane that is close to the runway
        runway_center_x, runway_center_y = 250, 205
        dx = plane.x - runway_center_x
        dy = plane.y - runway_center_y
        current_pos = dx**2 + dy**2
        if not hasattr(plane, 'pre_pos'):
            plane.pre_pos = current_pos
        movement = plane.pre_pos - current_pos
        reward += 0.005 * movement 
        plane.pre_pos = current_pos

        flight_state = int(obs[4])
        prev_state = plane.flight_state
        
        # Avoid the plane stay in the air for too long
        if plane.flight_state == 0 and self.time_step % 10 == 0:
            reward -= 0.5 

        if action in [0, 1]:  # turn actions
            if plane.flight_state == 0:
                reward -= 2  
            else:
                reward += self.move(plane, action)
        elif action in [2, 3, 12]:  # speed up/down or go_straight
            reward += self.move(plane, action)


        elif action in [4, 5, 6, 7]:  # Assign runway
            if plane.flight_state != 0:
                reward -= 1
            else:
                plane.runway = 0 if action in [4, 5] else 1
                plane.flight_state = 2

                # Set heading toward the assigned runway entry
                plane.entry_target = self.runway_entries[action]
                plane.runway_exit = self.runway_exits[action]
                plane.set_direction(*plane.entry_target)

                # Check alignment with wind
                runway_angle = {
                    4: np.pi / 2,
                    5: np.pi / 2,
                    6: 0,
                    7: 0
                }[action]
                aligned = self.is_within_pi(self.wind_direction, runway_angle)

                # Check if assignment matches recommended
                best_runway, best_angle = self.find_entry(plane)
                correct_assignment = (
                    plane.runway == best_runway and self.is_within_pi(runway_angle, best_angle)
                )

                if correct_assignment:
                    reward += 2
                else:
                    reward -= 1

                if obs[2] == 1 and action == 4:
                    reward -= 2  # Large plane 
                elif aligned:
                    reward += 4
                else:
                    reward -= 2

                reward += 1  # Base reward for valid assignment

                # Assign future gate based on runway direction
                plane.gate_target = self.gate_assign(action)

        elif action == 8:
            if plane.flight_state == 2: 
                reward += 2  
            elif plane.flight_state == 0:  # Still flying
                reward -= 1  
            else:
                reward += 0.5  

            plane.flight_state = 1  # taxiing
            plane.runway = None

            if self.is_on_taxiway(plane):
                reward += 0.5

        elif action == 9:
            if self.is_at_gate(plane):
                if plane.flight_state == 1:  # Taxiing to gate
                    reward += 5  # Full proper flow: land → taxi → gate
                elif plane.flight_state == 2:  # Skipped taxi
                    reward += 2  
                else:
                    reward -= 1  # Invalid gate attempt

                plane.flight_state = 3  
                self.planes.remove(plane)
                self.total_planes -= 1
                if self.total_planes < self.max_aircraft:
                    self.add_plane()

            else:
                reward -= 1  # Tried gate too early or missed location

            plane.runway = None
        
        # Completing the routine: landed → taxi → gate
        if plane.flight_state == 3 and prev_state == 1 and self.is_at_gate(plane):
            reward += 5

        elif action == 10: 
            if plane.flight_state == 1 and self.runway_occupied:
                reward += 1  # Waiting to access runway while taxiing

            close_to_others = any(
                other != plane and other.flight_state in [1, 2] and plane.distance_to(other.x, other.y) < 50
                for other in self.planes
            )
            if plane.flight_state == 1 and close_to_others and self.is_on_taxiway(plane):
                reward += 1.5  # yielding on taxiway
            
            elif plane.flight_state == 1 and close_to_others:
                reward += 1 # # yielding not on taxiway

            elif plane.flight_state == 2:
                aligned = self.is_within_pi(self.wind_direction, plane.direction)  # Lined up on runway and waiting for takeoff
                if aligned:
                    reward += 2  # Ready and aligned
                else:
                    reward += 0.5  # Waiting but needs to turn

            elif plane.flight_state in [1, 2]: # Avoid too much waitting
                reward -= 0.2 

            elif plane.flight_state == 0: # Wait in the air
                reward -= 1  

            elif plane.flight_state == 3: # Waiting at gate
                reward -= 0.5  

        elif action == 11:  
            if plane.flight_state == 2:
                aligned = self.is_within_pi(self.wind_direction, plane.direction)
                
                too_close = any(
                    other != plane and other.flight_state == 2 and plane.distance_to(other.x, other.y) < 50
                    for other in self.planes
                )

                if aligned and not too_close:
                    reward += 5 # ideal take off
                    plane.flight_state = 3 
                    self.planes.remove(plane)
                    self.total_planes -= 1
                    if self.total_planes < self.max_aircraft:
                        self.add_plane()
                
                elif aligned and too_close:
                    reward -= 2
                    print("Warning - too close to another plane.")

                elif not aligned:
                    reward -= 3 

            else:
                reward -= 2 # Take off when the plane is not on runway or the runway is not occupied

        plane.move()

        # Landing check
        if flight_state == 0 and self.is_on_runway(plane):
            if not self.runway_occupied:
                self.runway_occupied = True
                correct_angle = self.is_within_pi(self.wind_direction, plane.direction)
                if correct_angle:
                    reward += 100
                    plane.flight_state = 2
                else:
                    reward -= 50
            else:
                # Runway busy, move to taxiway
                plane.flight_state = 1
                plane.runway = None
                reward -= 20

        if (plane.x < -10 or plane.x > self.screen_width + 10 or
            plane.y < -10 or plane.y > self.screen_height + 10):
            reward -= 200
            done = True

        if random.random() < 0.03:
            reward -= 1000
            done = True

        self.state = [
            int(obs[2]),  # size
            int(obs[3]),  # speed bucket
            int(obs[4]),  # state
            plane.runway if plane.runway is not None else 0,
            self.wind_speed,
            self.wind_direction,
            1 if self.runway_occupied else 0
        ]

        if self.total_planes >= self.max_aircraft:
            done = True

        return np.array(self.state, dtype=np.float32), reward, done

    def step(self, actions):
        total_reward = 0
        done = False

        for idx, plane in enumerate(self.planes):
            action = actions[idx] if idx < len(actions) else 12
            plane.action = action
            if plane.flight_state == 0 and hasattr(plane, 'entry_target'):
                plane.set_direction(*plane.entry_target)
                dist = math.hypot(plane.x - plane.entry_target[0], plane.y - plane.entry_target[1])
                if dist < 20:
                    plane.flight_state = 2  # now on the runway
                    if hasattr(plane, 'runway_exit'):
                        plane.set_direction(*plane.runway_exit)
                    #print(f"Plane {idx} reached entry and switched to runway")

            _, reward, d = self.execute_action(plane, action)

            total_reward += reward
            done = done or d

            if plane.flight_state == 0:
                plane.move() 
            elif action in [8, 9, 11, 12]:  
                plane.move()
        
        collision = self.check_collisions()
        if collision:
            #print("Collision detected!")
            done = True  
            total_reward -= 100
        
        # Clean up planes
        self.planes = [p for p in self.planes if p.flight_state != 3]
        self.total_planes = len(self.planes)
        return np.array(self.state, dtype=np.float32), total_reward, done, {"collision": collision}

    def add_plane(self):
        entry_edge = random.choice(["top", "left"])
        entry_key = 4 if entry_edge == "top" else 6
        entry = self.runway_entries[entry_key]
        exit = self.runway_exits[entry_key]
        plane = Aircraft(self.screen_width, self.screen_height, entry_edge=entry_edge, entry_target=entry, runway_exit=exit)
        plane.entry_target = entry
        plane.runway_exit = exit
        self.planes.append(plane)
        self.total_planes += 1

    def render(self, mode='human'):
        print(f"Step: {self.time_step} | Wind: {self.wind_speed}, Dir: {self.wind_direction:.2f}, RunwayOccupied: {self.runway_occupied}")
        for i, p in enumerate(self.planes):
            print(f"Plane {i}: x={p.x:.1f}, y={p.y:.1f}, speed={p.speed:.1f}, state={p.flight_state}, runway={p.runway}")

