import numpy as np
import matplotlib.pyplot as plt

import nengo
from SBG import SBG
from nengo.dists import Uniform

############################################################################
# Bilateral
###########################################################################


class SBG_Pair(nengo.Network):

    def __init__(self, include_OPN=True, n_neurons=100, EBN_gain=1, TN_gain=1,
                 rec_tau=0.1, label=None, seed=None, add_to_container=None):
        """
        (OPTIONAL):
        neurons: number of neurons per ensemble
        inh_weights: the negative weight on inhibitory connections
        """

        super(SBG_Pair, self).__init__(label, seed, add_to_container)

        with self:
            self.left = SBG(n_neurons=n_neurons, EBN_gain=EBN_gain,
                            TN_gain=TN_gain)
            self.right = SBG(n_neurons=n_neurons, EBN_gain=EBN_gain,
                             TN_gain=TN_gain)

            for ens in self.left.ensembles:
                ens.label += '_l'
            for ens in self.right.ensembles:
                ens.label += '_r'

            if include_OPN:

                def neg(x):
                    return -10

                ''' pop thats always inhibiting, but slows down to allow saccade
                execution. Tuning curves are chosen s.t. cells have high
                firing rate when encoding 0'''
                self.OPN = nengo.Ensemble(n_neurons, dimensions=1,
                                          encoders=[[1]] * n_neurons,
                                          intercepts=Uniform(0, 0.1),
                                          radius=90,
                                          max_rates=Uniform(150, 200))

                # connections from OPN to EBN of both sides, OPN are shared
                nengo.Connection(self.OPN, self.left.EBN, function=neg)
                nengo.Connection(self.OPN, self.right.EBN, function=neg)
                nengo.Connection(self.OPN, self.left.IBN, function=neg)
                nengo.Connection(self.OPN, self.right.IBN, function=neg)

                # OPN inhibits LLBN
                nengo.Connection(self.OPN, self.left.LLBN, function=neg)
                nengo.Connection(self.OPN, self.right.LLBN, function=neg)

                # LLBN to OPN
                nengo.Connection(self.left.LLBN, self.OPN, function=neg)

                nengo.Connection(self.right.LLBN, self.OPN, function=neg)

            inh_weight = -0.0001

            # one side inhibits the other
            nengo.Connection(self.left.IBN, self.right.EBN, transform=-1)
            nengo.Connection(self.left.IBN.neurons, self.right.AMN.neurons,
                             transform=np.ones((n_neurons, n_neurons))*inh_weight)
            nengo.Connection(self.left.IBN.neurons, self.left.OMN.neurons,
                             transform=np.ones((n_neurons, n_neurons))*inh_weight)
            nengo.Connection(self.left.IBN, self.right.LLBN, transform=-1)

            nengo.Connection(self.right.IBN, self.left.EBN, transform=-1)
            nengo.Connection(self.right.IBN.neurons, self.left.AMN.neurons,
                             transform=np.ones((n_neurons, n_neurons))*inh_weight)
            nengo.Connection(self.right.IBN.neurons, self.right.OMN.neurons,
                             transform=np.ones((n_neurons, n_neurons))*inh_weight)
            nengo.Connection(self.right.IBN, self.left.LLBN, transform=-1)

            # MVN for computing absolute eye position in TN
            nengo.Connection(self.right.MVN, self.left.TN, transform=-1)
            nengo.Connection(self.left.MVN, self.right.TN, transform=-1)

            # AMN to OMN crossing
            nengo.Connection(self.left.AMN, self.right.OMN)
            nengo.Connection(self.right.AMN, self.left.OMN)
