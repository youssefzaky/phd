from __future__ import division
import numpy
from control.plants.linear_DS import Linear_DS, discretized_matrices

class Eye(Linear_DS):
    """Eye model is taken from 'Biological learning and control' by
    Shadmehr, pg """

    def __init__(self, dt, init_state=[[0], [0], [0], [0]], delay=0, Q=None, R=None):

        tau_1, tau_2 = 0.224, 0.013
        m, b = tau_1 * tau_2, tau_1 + tau_2
        self.m, self.k, self.b = m, 1, b
        alpha_1, alpha_2 = 0.004, 1
        self.alpha_1, self.alpha_2 = alpha_1, alpha_2

        # the continuous matrices
        self.c_A = numpy.array([[0, 1, 0, 0], [-1/m, -b/m, 1/m, -1/m],
                         [0, 0, -alpha_2/alpha_1, 0],
                         [0, 0, 0, -alpha_2/alpha_1]])
        self.c_B = numpy.array([[0, 0], [0, 0], [1/alpha_1, 0], [0, 1/alpha_1]])

        C = numpy.eye(self.c_A.shape[0])
        self.A, self.B = discretized_matrices(self.c_A, self.c_B, dt)
        super(Eye, self).__init__(self.A, self.B, C, Q=Q, R=R,
                                  init_state=init_state, delay=delay)
