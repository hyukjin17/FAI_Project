import math

class Taxiway:
    def __init__(self, x_start, y_start, x_end, y_end, name):
        """
        A simple Taxiway class for aircraft to move between runways and gates.

        Args:
            x_start (float): Starting x-coordinate
            y_start (float): Starting y-coordinate
            x_end (float): Ending x-coordinate
            y_end (float): Ending y-coordinate
            name (str): Optional name of the taxiway
        """
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.name = name
        self.occupied = False  # Whether a plane is currently using the taxiway

    def occupy(self):
        """Mark the taxiway as occupied."""
        self.occupied = True

    def free(self):
        """Mark the taxiway as free."""
        self.occupied = False

    def is_occupied(self):
        """Check if the taxiway is currently occupied."""
        return self.occupied
    
    def get_center(self):
        """Get the center coordinates of the taxiway."""
        center_x = (self.x_start + self.x_end) / 2
        center_y = (self.y_start + self.y_end) / 2
        return center_x, center_y
    
    def distance_to_center(self, x, y):
        """Get the distance from a given point to the center of the taxiway."""
        center_x, center_y = self.get_center()
        return math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)

    def length(self):
        """Return the length of the taxiway (straight line)."""
        if (self.x_end - self.x_start) > (self.y_end - self.y_start):
            length = self.x_end - self.x_start
        else:
            length = self.y_end - self.y_start
        return length

    def close_to(self, x, y, margin=5):
        """Check if a point (x, y) is close enough to the taxiway line."""
        # Simplified as a bounding box check
        min_x = min(self.x_start, self.x_end) - margin
        max_x = max(self.x_start, self.x_end) + margin
        min_y = min(self.y_start, self.y_end) - margin
        max_y = max(self.y_start, self.y_end) + margin

        return min_x <= x <= max_x and min_y <= y <= max_y