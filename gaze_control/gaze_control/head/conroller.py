import numpy

class OFC_EH(object):
    """Optimal feedback control for eye-head system"""

    def __init__(self, plant, c1, c2):
        self.plant = plant
        self.C1 = numpy.array([[c1, 0], [0, 0]])
        self.C2 = numpy.array([[0, 0], [0, c2]])
        self.L = numpy.diag(numpy.ones(2))

    def gain_sequence(self, duration, T):
        W_x = T
        W_e = 0
        G = []

        for i in range(duration):
            G.insert(0, self.compute_gain(W_x, W_e))
            W_x, W_e = self.udpate_matrices(G[0], W_x, W_e, K[i])

    def compute_gain(self, W_x, W_e):
        B = numpy.matrix(self.plant.B)
        C1, C2 = numpy.matrix(self.C1), numpy.matrix(self.C2)
        C_x = (C1.T * B.T * W_x * B * C1) + (C2.T * B.T * W_x * B * C2)
        C_e = (C1.T * B.T * W_e * B * C1) + (C2.T * B.T * W_x * B * C2)
        temp = L + C_x + C_e + B.T * W_x * B
        G = numpy.linalg.inv(temp) * B.T * W_x * self.plant.A
        return G

    def update_matrices(self, G, W_x, W_e, K):
        T_k = 0 # cost only at final step
        D_e = 0 # sig-dep noise on measurement eq. is 0
        A = numpy.matrix(self.plant.A)
        H = numpy.matrix(self.plant.C)
        W_x = T_k + D_e + (A.T * W_x * A) - (G.T * B.T * W_x * A)
        W_e = (A - A * K * H).T * W_e * (A - A * K * H) + (G.T * G.T * W_x * A)
