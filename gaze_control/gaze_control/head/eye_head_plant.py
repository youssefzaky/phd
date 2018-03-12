import sys
sys.path.append('/home/youssef/GitHub/eye')

import numpy

import eye

from control.plants.linear_DS import Linear_DS, discretized_matrices
from eye.eye_plant import Eye

class Head(Linear_DS):
    """Head model is taken from 'Biological learning and control' by
    Shadmehr, pg 358"""
    def __init__(self, dt, init_state=[0, 0, 0], delay=0, Q=None, R=None):
        tau_1, tau_2 = 0.270, 0.015
        m, b = tau_1 * tau_2, tau_1 + tau_2
        alpha_1, alpha_2 = 0.01, 1
        A = numpy.array([[0, 1, 0], [-1/m, -b/m, 1/m], [0, 0, -alpha_2/alpha_1]])
        B = numpy.array([[0], [0], [1/alpha_1]])
        C = numpy.eye(3)
        A, B = discretized_matrices(A, B, dt)
        super(Head, self).__init__(A, B, C, Q=Q, R=R,
                                   init_state=init_state, delay=delay)

    def effector_pos(self):
        return numpy.array([self.state[0], 0, 0])


class Eye_Head(Linear_DS):

    def __init__(self, dt, init_state=numpy.zeros(7), delay=0, Q=None, R=None):
        head = Head(dt)
        eye = Eye(dt)

        from scipy.linalg import block_diag

        A = block_diag(eye.A, head.A, [1])
        B = numpy.zeros((7, 2))
        B[:3, 0], B[3:7, 1] = eye.B, head.B
        print B

        C = [[-1, 0, 0, -1, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0]]

        Q = numpy.diag(numpy.ones(7))
        R = numpy.diag(numpy.ones(2))

        super(Eye_Head, self).__init__(A, B, C, Q=Q, R=R,
                                       init_state=init_state, delay=delay)
