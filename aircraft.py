import random
import numpy as np
import math

# Number to divide the speed of the aircraft by to scale it down
SPEED_FRACTION = 100

class Aircraft:
    PLANE_TYPES = {
        # Can change the type and speed range to more accurately represent the aircraft types
        "commercial": {"speed_range": (200, 300), "size": 7, "color": (0, 255, 0)},
        "cargo": {"speed_range": (150, 250), "size": 10, "color": (255, 165, 0)},
        "private": {"speed_range": (250, 350), "size": 5, "color": (0, 0, 255)},
        "military": {"speed_range": (300, 400), "size": 8, "color": (255, 0, 0)},
        "small": {"speed_range": (100, 200), "size": 2, "color": (0, 0, 0)}
    }

    def __init__(self, screen_width, screen_height):
        # Initialize a plane in air
        self.flight_state = 0   # 0: In Air, 1: Taxiway, 2: Runway, 3: At Gate
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Select a random plane type
        self.plane_type = random.choice(list(self.PLANE_TYPES.keys()))
        type_info = self.PLANE_TYPES[self.plane_type]

        # Assign speed based on type
        self.speed = random.uniform(type_info["speed_range"][0], type_info["speed_range"][1])
        self.max_speed = type_info["speed_range"][1]
        self.min_speed = type_info["speed_range"][0]
        self.size = type_info["size"]
        self.color = type_info["color"]
        self.direction = 0
        self.runway = None
        self.distance_to_runway = None
        self.taking_off = False
        self.turning_radius = self.update_turning_radius()

        if self.flight_state == 0:
            # Randomly spawn outside the screen
            # Set the angle of entry into the screen based on entry position
            edge = random.choice(["left", "right", "top", "bottom"])
            if edge == "left":
                self.x, self.y = -10, random.randint(0, screen_height)
                if self.y < screen_height / 2:
                    self.direction = random.uniform(0, np.pi / 4)
                else:
                    self.direction = random.uniform(7*np.pi / 4, 2*np.pi)
            elif edge == "right":
                self.x, self.y = screen_width + 10, random.randint(0, screen_height)
                if self.y < screen_height / 2:
                    self.direction = random.uniform(3 * np.pi / 4, np.pi)
                else:
                    self.direction = random.uniform(np.pi, 5 * np.pi / 4)
            elif edge == "top":
                self.x, self.y = random.randint(0, screen_width), -10
                if self.x < screen_width / 2:
                    self.direction = random.uniform(np.pi / 4, np.pi / 2)
                else:
                    self.direction = random.uniform(np.pi / 2, 3 * np.pi / 4)
            else:  # "bottom"
                self.x, self.y = random.randint(0, screen_width), screen_height + 10
                if self.x < screen_height / 2:
                    self.direction = random.uniform(3 * np.pi / 2, 7 * np.pi / 4)
                else:
                    self.direction = random.uniform(5 * np.pi / 4, 3 * np.pi / 2)
        else:
            # Set to the position of the airport
            self.x, self.y = 50, 50

        # Compute velocity components
        self.dx = (self.speed / SPEED_FRACTION) * math.cos(self.direction) if self.flight_state == 0 else 0 # Normalize speed
        self.dy = (self.speed / SPEED_FRACTION) * math.sin(self.direction) if self.flight_state == 0 else 0

    # Move by the speed at each time step
    def move(self):
        self.x += self.dx
        self.y += self.dy

    # Update the plane's direction angle proportionally based on speed and turning radius.
    # Does not move the plane's position.
    # Args:
    #     turn_direction (str): "left" or "right"
    def turn(self, turn_direction):
        omega = self.speed / self.turning_radius
        dtheta = omega / 30 # assuming the time step is 1/30 (30 fps)
        if turn_direction == "left":
            self.change_direction(-dtheta)
        else: # turn right
            self.change_direction(dtheta)
        
    def set_dx_dy(self, speed, direction):
        self.dx = (speed / SPEED_FRACTION) * math.cos(direction)  # Normalize speed
        self.dy = (speed / SPEED_FRACTION) * math.sin(direction)
        
    # Limit speed in air between min and max values
    # At runway, taxiway or gate, the speed can get as low as 0
    def change_speed(self, dv):
        can_update_speed = (
            self.flight_state != 0 and (0 < self.speed + dv < self.max_speed)) or (
            self.min_speed < self.speed + dv < self.max_speed)
        if can_update_speed:
            self.speed += dv
            self.set_dx_dy(self.speed, self.direction)
            self.turning_radius = self.update_turning_radius()

    # Changes direction by a given angle and updates the dx and dy
    def change_direction(self, angle):
        self.direction = (self.direction + angle) % (2 * np.pi)
        self.set_dx_dy(self.speed, self.direction)

    # Changes direction to a given direction and updates the dx and dy
    def set_direction(self, direction):
        self.direction = direction
        self.set_dx_dy(self.speed, self.direction)

    # Check if the plane has exited the screen
    def is_off_screen(self):
        return (self.x < -10 or self.x > self.screen_width + 10 or
                self.y < -10 or self.y > self.screen_height + 10)
    
    # Set plane speed and velocity to 0.
    def stop(self):
        self.speed = 0
        self.dx = 0
        self.dy = 0

    # Calculate distance to another point.
    def distance_to(self, x, y):
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    # Check if another plane is within a threshold distance.
    def is_close_to(self, other_plane, threshold):
        return self.distance_to(other_plane.x, other_plane.y) < threshold
    
    # Update turning radius based on current speed.
    def update_turning_radius(self):
        return 2 * self.size * (self.speed / SPEED_FRACTION) ** 2
    
    def assign_runway(self, runway):
        self.runway = runway
        self.distance_to_runway = self.distance_to(runway.x_entry, runway.y_entry)