import numpy

class Environment(object):

    def __init__(self, plant):
        self.plant = plant

    def step(self, dt):
        raise NotImplementedError()
