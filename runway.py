import math

class Runway:
    def __init__(self, x_start, y_start, x_end, y_end, x_entry, y_entry, direction, name="Runway"):
        """
        Initialize a runway.

        Args:
            x_start (int): Starting x-coordinate of the runway.
            y_start (int): Starting y-coordinate of the runway.
            x_end (int): Ending x-coordinate of the runway.
            y_end (int): Ending y-coordinate of the runway.
            direction (float): Direction the runway is facing (in radians).
            name (str): Name or ID of the runway (default "Runway").
        """
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.x_entry = x_entry
        self.y_entry = y_entry
        self.direction = direction  # in radians
        self.name = name
        self.occupied = False  # Whether a plane is currently on the runway

    def occupy(self):
        """Mark the runway as occupied by an aircraft."""
        self.occupied = True

    def free(self):
        """Mark the runway as free."""
        self.occupied = False

    def is_occupied(self):
        """Return whether the runway is occupied."""
        return self.occupied

    def get_length(self):
        """Calculate the length of the runway."""
        if (self.x_end - self.x_start) > (self.y_end - self.y_start):
            length = self.x_end - self.x_start
        else:
            length = self.y_end - self.y_start
        return length
    
    def change_direction(self):
        """Change runway direction."""
        self.direction = (self.direction + math.pi) % (2 * math.pi)
        if math.isclose(self.direction, 0, abs_tol=0.001):
            self.x_entry = self.x_start
        elif math.isclose(self.direction, math.pi, abs_tol=0.001):
            self.x_entry = self.x_end
        elif math.isclose(self.direction, math.pi/2, abs_tol=0.001):
            self.y_entry = self.y_start
        elif math.isclose(self.direction, 3*math.pi/2, abs_tol=0.001):
            self.y_entry = self.y_end

    def on_runway(self, x, y):
        """Checks whether a point is on a runway"""
        return self.x_start < x < self.x_end and self.y_start < y < self.y_end

    def __repr__(self):
        return f"{self.name}: ({self.x_start},{self.y_start}) -> ({self.x_end},{self.y_end}), direction {self.direction:.2f} rad"