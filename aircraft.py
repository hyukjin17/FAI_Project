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
        # Initialize a plane randomly either in air or at the gate
        self.flight_state = random.choice([0,3])

        # Select a random plane type
        self.plane_type = random.choice(list(self.PLANE_TYPES.keys()))
        type_info = self.PLANE_TYPES[self.plane_type]

        # Assign speed based on type
        self.speed = random.uniform(type_info["speed_range"][0], type_info["speed_range"][1])
        self.max_speed = type_info["speed_range"][1]
        self.size = type_info["size"]
        self.color = type_info["color"]
        self.direction = 0
        self.runway = None
        self.takeoff = True if self.flight_state == 3 else False
        self.turning_radius = self.size * (self.speed / SPEED_FRACTION) * (self.speed / SPEED_FRACTION)

        if self.flight_state == 0:
            # Randomly spawn outside the screen
            # Set the angle of entry into the screen based on entry position
            edge = random.choice(["left", "right", "top", "bottom"])
            if edge == "left":
                self.x, self.y = -10, random.randint(0, screen_height)
                if self.y < screen_height / 2:
                    self.direction = random.uniform(0, np.pi / 4)
                else:
                    self.direction = random.uniform(2*np.pi, 7*np.pi / 4)
            elif edge == "right":
                self.x, self.y = screen_width + 10, random.randint(0, screen_height)
                if self.y < screen_height / 2:
                    self.direction = random.uniform(3 * np.pi / 4, np.pi)
                else:
                    self.direction = random.uniform(5 * np.pi / 4, np.pi)
            elif edge == "top":
                self.x, self.y = random.randint(0, screen_width), -10
                if self.x < screen_width / 2:
                    self.direction = random.uniform(3 * np.pi / 2, 7 * np.pi / 4)
                else:
                    self.direction = random.uniform(5 * np.pi / 4, 3 * np.pi / 2)
            else:  # "bottom"
                self.x, self.y = random.randint(0, screen_width), screen_height + 10
                if self.x < screen_height / 2:
                    self.direction = random.uniform(np.pi / 4, np.pi / 2)
                else:
                    self.direction = random.uniform(np.pi / 2, 3 * np.pi / 4)
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

    def turn(self, turn_direction):
        omega = self.speed / self.turning_radius
        dtheta = omega

        if (turn_direction == "left"):
            cx = self.x - self.turning_radius * math.sin(self.direction)
            cy = self.y + self.turning_radius * math.cos(self.direction)
            self.change_direction(dtheta)
        else:
            cx = self.x + self.turning_radius * math.sin(self.direction)
            cy = self.y - self.turning_radius * math.cos(self.direction)
            self.change_direction(-dtheta)
        self.x = cx + self.turning_radius * math.cos(self.direction)
        self.y = cy + self.turning_radius * math.sin(self.direction)
        
    def set_dx_dy(self, speed, direction):
        self.dx = (speed / SPEED_FRACTION) * math.cos(direction)  # Normalize speed
        self.dy = (speed / SPEED_FRACTION) * math.sin(direction)
        
    def change_speed(self, dv):
        if (self.speed + dv < self.max_speed):
            self.speed += dv
            self.set_dx_dy(self.speed, self.direction)
            self.turning_radius = self.size * (self.speed / SPEED_FRACTION) * (self.speed / SPEED_FRACTION)

    def change_direction(self, direction):
        if self.direction != direction:
            self.direction = (self.direction + direction) % (2 * np.pi)
            self.set_dx_dy(self.speed, self.direction)


    # Check if the plane has exited the screen
    def is_off_screen(self):
        return (self.x < -10 or self.x > self.screen_width + 10 or
                self.y < -10 or self.y > self.screen_height + 10)

    # Current state of the plane
    # May need to add more info (like nearby planes)
    def get_obs(self):
        plane_type = list(self.PLANE_TYPES.keys()).index(self.plane_type)  # Assign integer value (index) to plane type
        speed = 0 if self.speed < 200 else 1 if self.speed < 400 else 3
        return np.array([0 if self.size < 5 else 1, speed, plane_type, self.flight_state], dtype=np.float32)