"""This file contains the Estimator class."""

class Estimator(object):
    """Base class of all estimators."""

    def __init__(self, plant):
        self.plant = plant
        self.estimate = plant.state
        
    def step(self, control, meas, dt):
        """Simply gets the true state of the plant."""
        self.estimate = self.plant.state
