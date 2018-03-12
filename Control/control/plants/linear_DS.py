"""
This file contains the Linear_DS class.

TODO: how to use different integration methods
TODO: validate input
TODO: use descriptors to implement time-varying systems
      so that attributes like A,B.. 'get called'
"""

import numpy
from control.noise import Gaussian_Noise
from base_plant import Plant


def discretized_matrices(A, B, dt):
    """Helper function to make discrete versions of the A, B matrices
    when used in simulation. While this could be done in the simulation,
    it is necessary to assign them this way if they are
    used in other calculations such as optimal control laws."""
    return A * dt + numpy.eye(A.shape[0]), B * dt


class Linear_DS(Plant):
    """
    Implementation of a linear dynamical system plant with additive Gaussian noise.
    The state-space equations are

    xdot = Ax + Bu + w
    y = Cx + v

    NOTE: this assumes the plant matrices are discretized. Meaning that
    A = A_c * dt + I
    B = B_c * dt

    where A_c and B_c are the matrices of the continuous system.

    Parameters:

    A, B, C: matrices in the state-space equations
    Q: covariance of process noise
    R: covariance of measurement noise
    """

    def __init__(self, A, B, C, Q=None, R=None, init_state=None, delay=0):
        self.A = numpy.matrix(A)
        self.B = numpy.matrix(B)
        self.C = numpy.matrix(C)
        self.Q = Q
        self.R = R
        init_state = numpy.matrix(init_state)
        state_dim = A.shape[0]
        meas_dim = C.shape[0]

        if Q is not None:
            self.process_noise = Gaussian_Noise(Q)
        if R is not None:
            self.meas_noise = Gaussian_Noise(R)

        super(Linear_DS, self).__init__(state_dim, meas_dim, init_state, delay=delay)

    def system_model(self, state, control, dt):
        """
        Currently uses the Euler stepping method, if necessary
        we can add options for other methods.
        """

        if self.Q is None:
            state = self.A * state +  self.B * control
        else:
            state = numpy.array(self.A * state +  self.B * control).flatten()
            state = self.process_noise(state)
            state = numpy.matrix(state).T

        return state

    def meas_model(self, state):

        if self.R is None:
            meas = self.C * state
        else:
            meas = numpy.array(self.C * state).flatten()
            meas = self.meas_noise(meas)
            meas = numpy.matrix(meas).T

        return meas
