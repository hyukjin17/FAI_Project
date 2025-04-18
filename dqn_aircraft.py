import random
import numpy as np
import math

class Aircraft:
    PLANE_TYPES = {
        "commercial": {"speed_range": (200, 300), "size": 7, "color": (0, 255, 0)},
        "cargo": {"speed_range": (150, 250), "size": 10, "color": (255, 165, 0)},
        "private": {"speed_range": (250, 350), "size": 5, "color": (0, 0, 255)},
        "military": {"speed_range": (300, 400), "size": 8, "color": (255, 0, 0)},
        "small": {"speed_range": (100, 200), "size": 2, "color": (0, 0, 0)}
    }

    def __init__(self, screen_width, screen_height, entry_edge=None, entry_target=None, runway_exit=None):
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.plane_type = random.choice(list(self.PLANE_TYPES.keys()))
        type_info = self.PLANE_TYPES[self.plane_type]

        self.speed = random.uniform(*type_info["speed_range"])
        self.size = type_info["size"]
        self.color = type_info["color"]
        self.align = False  # Track alignment with wind

        SPEED_FRACTION = 100
        
        edge = entry_edge if entry_edge in ["top", "left"] else random.choice(["top", "left"])
        self.entry_edge = edge
        self.entry_target = entry_target  
        self.runway_exit = runway_exit
        
        # Spawn location and direction toward airport center
        if edge == "left":
            self.x = -10
            self.y = random.randint(100, screen_height - 100)
        else:  # "top"
            self.x = random.randint(100, screen_width - 100)
            self.y = -10


        # Direction aimed toward center of airport 
        target_x = screen_width // 2
        target_y = screen_height // 2
        dx = target_x - self.x
        dy = target_y - self.y
        self.direction = math.atan2(dy, dx)

        self.runway = None
        self.flight_state = 0

        self._update_velocity()

    def move(self):
        self.x += self.dx
        self.y += self.dy

    def turn(self, turn_direction):
        turn_angle = np.pi / 18  # 10 degrees

        if turn_direction == "left":
            self.direction += turn_angle
        elif turn_direction == "right":
            self.direction -= turn_angle

        self.direction = self.direction % (2 * np.pi)

        self._update_velocity() 

    def change_speed(self, delta):
        self.speed = max(0, self.speed + delta)
        self._update_velocity()

    def _update_velocity(self):
        SPEED_FRACTION = 100
        self.dx = (self.speed / SPEED_FRACTION) * math.cos(self.direction)
        self.dy = (self.speed / SPEED_FRACTION) * math.sin(self.direction)

    def get_obs(self):
        plane_type = list(self.PLANE_TYPES.keys()).index(self.plane_type)
        speed_bucket = 0 if self.speed < 200 else 1 if self.speed < 350 else 2
        size_class = 0 if self.size < 5 else 1  # small/large
        return np.array([size_class, speed_bucket, plane_type, self.flight_state], dtype=np.int32)

    def set_direction(self, target_x, target_y):
        self.direction = math.atan2(target_y - self.y, target_x - self.x)
        self._update_velocity()

    def distance_to(self, target_x, target_y):
        return math.hypot(self.x - target_x, self.y - target_y)