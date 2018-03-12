import numpy

class Controller(object):
    """Base class of all controllers.
    Simply outputs zeros as a control signal."""

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, estimate):
        return numpy.zeros(self.dim)

class Gravity(Controller):
    """
    Defines a gravity control force.

    Parameters:
    (Optional) gravity: force of gravity in m/s^2
    """

    def __init__(self, dim, gravity=-9.8):
        self.gravity = gravity
        self.control_signal = numpy.zeros(dim)
        self.control_signal[1] = gravity

    def __call__(self, estimate):
        return self.control_signal

class Composite(Controller):
    """
    Used to define multiple control signal that are added together.
    Note: control signals must be of same dimension

    Parameters:
    controllers: a list of all the controllers to be summed
    """

    def __init__(self, controllers):
        self.controllers = controllers

    def __call__(self, estimate):

        #Only works if we have more than one controller
        assert(len(self.controllers) > 1)

        control = self.controllers[0](estimate)
        for controller in self.controllers[1:] :
            control += controller(estimate)

        return control

class PID(Controller):

    def __init__(self, error_dim, control_dim, dt, K_p=1, K_d=1, K_i=1):
        self.signal = numpy.zeros(control_dim)
        self.K_p = K_p
        self.K_d = K_d
        self.K_i = K_i
        self.old_error = numpy.zeros(error_dim)
        self.i_error = numpy.zeros(error_dim)
        self.dt = dt

    def __call__(self, error):
        p = self.K_p * error
        d = self.K_d * (error - self.old_error) / self.dt
        self.old_error = error
        i = self.K_i * self.i_error
        self.i_error += error * self.dt
        return numpy.matrix(p + d + i)
