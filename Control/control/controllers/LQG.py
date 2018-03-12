import numpy

from control.estimators.kalman_filter import Kalman_Filter
from control.controllers.LQR import LQR

class LQG(object):

    def __init__(self, plant, cost, dt):
        self.plant = plant
        self.dt = dt
        self.LQR = LQR(plant, cost)
        self.KF = Kalman_Filter(plant)
        self._control = self.LQR.control(plant.state)

    def control(self):
        self.KF.step(self._control, self.plant.meas, self.dt)
        self._control = self.LQR.control(self.KF.estimate)
        return self._control
