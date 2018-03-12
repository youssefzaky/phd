import sys

import numpy

from eye_plant import Eye

from scipy.linalg import block_diag

# the following two function are used to store and retrieve
# the appropriate matrices specific to this eye problem
# This is done to avoid recomputing these

def compute_matrices(max_dur, dt):

    eye = Eye(dt=dt)

    # make F and G matrices as in the book
    F = [numpy.eye(eye.state_dim)]
    for _ in range(max_dur - 1):
        #add to front of list
        F.insert(0, numpy.dot(F[0], eye.A))

    # block matrix
    F = numpy.bmat(F)

    # block diagonal matrix
    G = block_diag(*[eye.B for _ in range(max_dur)])

    numpy.savez('matrices', F=F, G=G)


def get_matrices(duration):
    state_dim = 4
    control_dim = 2
    data = numpy.load('matrices.npz')
    F = data['F']
    G = data['G']
    F = F[:, (-state_dim * duration):]
    G = G[:(state_dim * duration), :control_dim * duration]

    return F, G


# if this file is run, the matrices will be computed
if __name__ == '__main__':
    dt = float(sys.argv[1])
    compute_matrices(max_dur=300, dt=dt)


class Optimal_Eye(object):
    """The optimal eye controller in section 11.4 of Shadmehr book
      Signal-Dependent noise version"""

    def __init__(self, plant):
        #need plant and cost to construct the controller
        self.plant = plant

    def control_seq(self, init_state, target, k):
        """Computes control sequence for the assigned duration.

        Parameters:
        -----------

        init_state: initial_state of the system
        target: the goal target, see pg.318
        k: the parameter in signal-dep noise on control
        """

        # compute the duration from experimental data
        # see pg 285 in Shadmehr book
        distance = numpy.abs(target - init_state[0][0])
        duration = int(round(2.7 * distance + 23))

        # cost matrices, see section 11.4
        T = numpy.diag([5e9, 1e6, 80, 80])
        L = numpy.eye(duration * self.plant.B.shape[1])

        # set up matrices
        L = numpy.matrix(L, dtype=numpy.float64)
        C = numpy.matrix(self.plant.C, dtype=numpy.float64)
        F, G = get_matrices(duration)
        F = numpy.matrix(F, dtype=numpy.float64)
        G = numpy.matrix(G, dtype=numpy.float64)
        T = numpy.matrix(T, dtype=numpy.float64)
        # A to the power of duration
        A_p = numpy.dot(self.plant.A, F[:, :self.plant.state_dim])
        A_p = numpy.matrix(A_p, dtype=numpy.float64)

        S = G.T * F.T * C.T * T * C * F * G
        diag_S = numpy.diag(numpy.diag(S))
        temp = L + S + ((k**2) * diag_S)
        temp = numpy.linalg.inv(temp)
        temp = temp * G.T * F.T * C.T * T
        temp = temp * (target - C * A_p * init_state)

        control = temp
        return control
