import numpy
from control.noise import Noise

"""TODO: fix the weightings with the paramter c.
         Make it a convex combination?"""


class Forward(object):
    """Base class of forward models.

    Parameters:
    (Optional) future_steps: state and meas will be predicted at time
                             future_steps * dt from the current time
    (Optional) noise: noise class, noise is added to the initial state and meas, and is
                      propagated to future predictions
    (Optional) c: the constant that's used for the averaging (1/c)
    """

    def __init__(self, state_dim, meas_dim, system_model, meas_model, dt,
                 future_steps=10, noise=Noise(), c=1000):
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.system_model = system_model
        self.meas_model = meas_model
        self.future_steps = future_steps
        self.dt = dt
        self.Noise = noise
        self.state_window = numpy.zeros((state_dim, future_steps))
        self.meas_window = numpy.zeros((meas_dim, future_steps))
        self.c = c

    def step(self, state, control):

        state_array = numpy.empty((self.state_dim, self.future_steps + 1))
        meas_array = numpy.empty((self.meas_dim, self.future_steps + 1))

        #set first elements of arrays to initial state, meas plus noise
        #the noise propagates through future estimates
        state_array[:, 0] = self.Noise(state)
        meas_array[:, 0] = self.Noise(self.meas_model(state))

        control = self.pad_control(control)

        for i in range(1, self.future_steps + 1):
            state_array[:, i] = self.system_model(state_array[:, i-1],
                                                  control[:, i-1], self.dt)
            meas_array[:, i] = self.meas_model(state_array[:, i])

        #shift windows by one step and append zeros
        self.state_window[:, :-1] = self.state_window[:, 1:]
        self.meas_window[:, :-1] = self.meas_window[:, 1:]

        self.state_window[:, -1] = numpy.zeros(self.state_dim)
        self.meas_window[:, -1] = numpy.zeros(self.meas_dim)

        #average new predicions with old one
        self.state_window = state_array[:, 1:] + (1./self.c) * self.state_window
        self.meas_window = meas_array[:, 1:] + (1./self.c) * self.meas_window

    def pad_control(self, control):
        #make control signal a matrix if its a 1D array
        if len(control.shape) < 2:
            control = control.reshape((control.shape[0], 1))

        #TODO: test this for control signals that are longer than
        #      one timestep

        #pad the control with its last element
        #to make it align with number of future steps
        #Note: can take a control signal for multiple timesteps
        if control.shape[1] < self.future_steps:
            temp = control
            control_size = control.shape[1]
            control_dim = control.shape[0]
            control = numpy.empty((control_dim, self.future_steps))

            for i in range(control_size):
                control[:, i] = temp[:, i]
            for i in range(control_size, self.future_steps):
                control[:, i] = temp[:,-1]

        return control
