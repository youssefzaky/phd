import numpy as np

class LQR(object):
    """Computes the inifinte horizon, discrete-time LQR optimal control law"""

    def __init__(self, plant, cost):
        self.R = np.matrix(cost.L)
        self.Q = np.matrix(cost.Q)
        self.A = np.matrix(plant.A)
        self.B = np.matrix(plant.B)
        self.P = np.matrix(cost.T)
        self.gain = self.get_gain(self.get_P())

    def get_gain(self, P):
        """get the optimal gain"""
        return -np.linalg.inv(self.R + self.B.T * P * self.B) * self.B.T * P * self.A

    def get_P(self):
        """Iterate the algebraic Ricatti equation until convergence"""

        P = self.P

        while True:
            new_P = self.Q + self.A.T * \
                (P - P * self.B * \
                     np.linalg.inv(self.R + self.B.T * P * self.B) * \
                     self.B.T * P) * self.A

            if np.linalg.norm(P - new_P) < 0.001:
                return new_P
            else:
                P = new_P

    def control(self, state):
        return self.gain * state
