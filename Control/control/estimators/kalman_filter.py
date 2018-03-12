#coding: utf-8
"""This file contains the Kalman_Filter class."""

from numpy import dot, eye
import numpy
from numpy.linalg import inv

from base_estimator import Estimator


"""
Abbreviations:

X: mean state estimate of the previous step (k − 1).
P: state covariance of previous step (k − 1).
A: transition n x n matrix.
Q: process noise covariance matrix.
B: input effect matrix.
C: measurement model matrix
U: control input.
K: Kalman Gain matrix
IC: innovation covariance
"""

class Kalman_Filter(Estimator):
    """
    Implements the kalman filter.

    Parameters:
    system: a linear dynamical system
    """

    def __init__(self, system):
        #initialize state and covariance estimates
        self.P = eye(system.state_dim) * 1000000
        self.X = system.state
        self.estimate = self.X
        self.K = None
        super(Kalman_Filter, self).__init__(system)

    def predict(self, X, P, U, dt):
        """ Implements the following equations:
        X = AX + BU
        P = APA_T + Q
        """
        A = self.plant.A
        X = X + (self.plant.A * X + self.plant.B * U) * dt
        P = A * P * A.T + self.plant.Q
        return (X, P)

    def update(self, X, P, Y):
        C = self.plant.C
        IC = C * P * C.T + self.plant.R
        K = P * C.T * inv(IC)
        X = X + K * (Y - C * X)
        P = P - K * C * P
        self.K = K
        return (X, P)

    def step(self, control, meas, dt):
        (self.X, self.P) = self.update(self.X, self.P, meas)
        (self.X, self.P) = self.predict(self.X, self.P, control, dt)
        self.estimate = self.X
