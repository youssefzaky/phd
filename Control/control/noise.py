import numpy


class Noise(object):
    """Base class of noise classes.
       Doesn't add any noise."""

    def __call__(self, var):
        return var


class Gaussian_Noise(Noise):
    """Additive gaussian noise with mean 0 and covariance matrix cov"""
    def __init__(self, cov):
        self.cov = cov

    def __call__(self, var):
        return numpy.random.multivariate_normal(var, self.cov)


class Signal_Dep_Noise(Noise):
    """
    Adds signal dependent noise to a signal:
    signal_i * (1 + c_i * phi_i), where phi_i is N(0,1)
    """

    def __init__(self, c):
        """c is an array of the noise coefficients for signal dep noise"""
        self.c = numpy.asarray(c)

    def __call__(self, signal):
        signal = numpy.array(signal)
        noise = self.c * signal * numpy.random.randn(*signal.shape)
        return numpy.matrix(signal + noise)
