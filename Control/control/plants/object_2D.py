from linear_DS import Linear_DS, discretized_matrices
import numpy

class Object_2D(Linear_DS):
    """Point object moving in 2D.

    Parameters:
    init_state: [x, y, dx, dy].
                where (x,y) are the position and (dx,dy) are the velocities.
    M: mass of object, kg
    """

    def __init__(self, dt, init_state = [0, 0, 0, 0], M=3, delay=0, Q=None, R=None):
        self.M = M

        A = numpy.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        B = numpy.array([[0, 0], [0, 0], [0, 0], [0, self.M]])
        C = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        A, B = discretized_matrices(A, B, dt)
        super(Object_2D, self).__init__(A, B, C, Q=Q, R=R, init_state=init_state, delay=delay)
