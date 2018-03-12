import nengo

class SBG(nengo.Network):

    def __init__(self, n_neurons=200, pulse_rad=200, con_rad=150, int_rad=45,
                 EBN_gain=8, TN_gain=30, IBN_gain=-1, rec_tau=0.005,
                 label=None, seed=None, add_to_container=None):

        super(SBG, self).__init__(label, seed, add_to_container)

        with self:
            self.config[nengo.Ensemble].encoders = nengo.dists.Uniform(1, 1)
            self.config[nengo.Ensemble].intercepts = nengo.dists.Uniform(0, 0.9)

            # LLBN encodes the motor error
            self.LLBN = nengo.Ensemble(n_neurons=n_neurons, dimensions=1,
                                       label='LLBN', radius=int_rad)

            # EBN outputs pulse term
            self.EBN = nengo.Ensemble(n_neurons, dimensions=1, label='EBN',
                                      radius=pulse_rad)

            # IBN outputs derivative term
            self.IBN = nengo.Ensemble(n_neurons, dimensions=1, label='IBN',
                                      radius=pulse_rad,
                                      encoders=nengo.dists.Uniform(-1, 1),
                                      intercepts=nengo.dists.Uniform(-1, 1))

            # Abducens MN: PID control signal
            self.AMN = nengo.Ensemble(n_neurons, dimensions=1, label='AMN',
                                      radius=con_rad)

            # oculomotor MN: sum of pulse and step, position and velocity
            self.OMN = nengo.Ensemble(n_neurons, dimensions=1, label='OMN',
                                      radius=con_rad)

            # TN outputs integrator term
            self.TN = nengo.Ensemble(n_neurons=400, dimensions=1, label='TN',
                                     radius=int_rad)

            # connections
            nengo.Connection(self.LLBN, self.EBN)
            nengo.Connection(self.LLBN, self.TN, transform=TN_gain * rec_tau)
            nengo.Connection(self.EBN, self.IBN, transform=EBN_gain)
            self.EBN_AMN = nengo.Connection(self.EBN, self.AMN,
                                            transform=EBN_gain)
            self.IBN_OMN = nengo.Connection(self.IBN, self.OMN,
                                            transform=IBN_gain)
            nengo.Connection(self.TN, self.TN, synapse=rec_tau)
            self.TN_AMN = nengo.Connection(self.TN, self.AMN)
