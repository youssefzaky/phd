"""This file contains the Plant class.

TODO: add ability to do multiple steps with same control or with longer control"""

import numpy
import collections

class Plant(object):
    """
    Base class of all plants. Plants are dynamical systems
    in state-space formulation.

    Parameters:

    state_dim: dimension of the state vector
    meas_dim: dimension of the measurement vector
    (Optional) init_state: initial state of the plant, zero otherwise
    """

    def __init__(self, state_dim, meas_dim, init_state, delay=0):
        """Initialize dimensions, state and measurement."""

        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.init_state = init_state

        if init_state is None:
            self.state = numpy.matrix(numpy.zeros(self.state_dim))
        else:
            self.state = numpy.matrix(init_state)

        self.delay = delay
        self.meas_queue = collections.deque(maxlen=delay + 1)
        self.meas = self.meas_model(self.state)

    def meas_model(self, state):
        """Implements the measurement model of the dynamical system."""
        raise NotImplementedError()

    def system_model(self, state, control, dt):
        """Implements the system model of the dynamical system."""
        raise NotImplementedError()

    def step(self, control, dt):
        """Step forward in time."""

        self.state = self.system_model(self.state, control, dt)
        self.meas_queue.append(self.meas_model(self.state))

    def randomize_state(self):
        """Set the plant to a random state."""
        raise NotImplementedError()

    def reset(self):
        """Zeros the state and measurements"""
        self.state = self.init_state
        self.meas = self.meas_model(self.state)

    @property
    def meas(self):
        return self.meas_queue[0]

    @meas.setter
    def meas(self, meas):
        self.meas_queue.clear()
        #initialize the queue with the first measurement
        for _ in range(self.delay + 1):
            self.meas_queue.append(meas)
