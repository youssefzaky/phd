from environment import Environment

class Box(Environment):
    """
    Defines an environment in the shape of a box.
    Bouncing is done by flipping the sign of x or y coordinate,
    
    Parameters:
    (Optional) x: length of box
    (Optional) y: height of box
    """
    
    def __init__(self, plant, x_left=-1, x_right=1, y_up=1, y_down=-1):
        self.x_left = x_left
        self.x_right = x_right
        self.y_up = y_up
        self.y_down = y_down
        super(Box, self).__init__(plant)

    def step(self, dt):
        """Assumes plant is 2D."""

        state = self.plant.state

        if state[0] <= self.x_left:
            state[0] = self.x_left
            state[2] *= -1
            state[2] -= .1 * state[2]

        if state[0] >= self.x_right:
            state[0] = self.x_right
            state[2] *= -1
            state[2] -= .1 * state[2]

        if state[1] >= self.y_up:
            state[1] = self.y_up
            state[3] *= -1
            state[3] -= .1 * state[3]

        if state[1] <= self.y_down:
            state[1] = self.y_down
            state[3] *= -1
            state[3] -= .1 * state[3]
        
