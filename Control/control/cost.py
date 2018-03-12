from __future__ import division
from numpy import dot

class Cost(object):
    def __init__(self, T=0, L=0, Q=0, alpha=0, beta=0):
        self.T = T
        self.Q = Q
        self.L = L # sometimes denoted by R
        self.alpha = alpha
        self.beta = beta

    def __call__(self, y_final, goal, control):
        accuracy_cost = dot(dot((y_final - goal).T,
                                 self.T), y_final - goal)
        motor_cost = dot(dot(control.T, self.L), control)
        time = control.shape[0]
        discount = self.alpha * (1 - 1/(1 + self.beta * time))
        return accuracy_cost + motor_cost + discount
