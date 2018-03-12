import numpy as np
from control.plants.linear_DS import Linear_DS, discretized_matrices

B_1 = 5.7223
B_2 = 0.5016
B = 0.327
J = 0.0022
r = 0.0118
K_se = 124.9582
K_lt = 60.6874
K = 16.3597
B_12 = B_1 + B_2
K_st = K_se + K_lt

delta = 5208.7 / (J * B_12)
P_2 = (J*K_st + B_12*B + 2*B_1*B_2) / (J*B_12)
P_1 = (2*B_1*K_se + 2*B_2*K_lt + B_12*K + K_st*B) / (J*B_12)
P_0 = (K_st*K + 2*K_lt*K_se) / (J*B_12)


class Eye(Linear_DS):
    """Eye model is taken from Enderle and Zhu"""

    def __init__(self, dt, init_state=[[0], [0], [0], [0]], delay=0,
                 Q=None, R=None):

        # the continuous matrices
        self.c_A = np.array([[0, 1, 0], [0, 0, 1],
                             [-P_0, -P_1, -P_2]])
        self.c_B = np.array([[0, 0, 0, 0], [0, 0, 0, 0],
                             [K_se, -K_se, B_2, -B_2]])

        C = np.eye(self.c_A.shape[0])
        self.A, self.B = discretized_matrices(self.c_A, self.c_B, dt)
        super(Eye, self).__init__(self.A, self.B, C, Q=Q, R=R,
                                  init_state=init_state, delay=delay)


t_gac = 0.0112
t_gde = 0.0065
t_tde = 0.0048
t_tac = 0.0093

# class F_ag(Linear_DS):
