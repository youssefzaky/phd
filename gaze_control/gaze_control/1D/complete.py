import numpy as np
import nengo
from SBG import SBG
from SC import SC
from internal_models import Internal_Model
import sys
sys.path.append('/home/youssef/GitHub/cbm')
from adf import cbm

######### Note: check tuning curves #############

class Target(object):
    def __init__(self, target=0, stim_start=0, stim_end=0):
        self.target = target
        self.stim_start = stim_start
        self.stim_end = stim_end

    def step(self, t):
        return self.target if t > self.stim_start and t < self.stim_end else 0


def make_network(eye, target, dt):
    # inh_weight = -0.001

    def plant_model(t, x):
        x = np.matrix(x).T
        eye.step(x, dt)
        if eye.state[2] > 250: eye.state[2] = 250
        return np.array(eye.state).squeeze()

    with nengo.Network() as model:
        model.config[nengo.Ensemble].encoders = nengo.dists.Uniform(1, 1)
        model.config[nengo.Ensemble].intercepts = nengo.dists.Uniform(0, 0.9)
        model.target = nengo.Node(output=target.step)
        model.SC_net = SC(n_neurons=1000)
        model.plant = nengo.Node(size_in=2, output=plant_model,
                                   label='l_eye')
        model.cbm = cbm()

        model.OPN = nengo.Ensemble(dimensions=1, n_neurons=200)
        model.internal = Internal_Model(eye, neurons=1000)
        model.SBG = SBG()

        nengo.Connection(model.target, model.cbm.MF, transform=1./45)
        nengo.Connection(model.SC_net.output[0], model.cbm.IO, transform=1./30)
        nengo.Connection(model.target, model.SC_net.stim_transform[0], synapse=None)
        nengo.Connection(model.SC_net.output[0], model.SBG.LLBN, synapse=None)
        # nengo.Connection(model.SBG.LLBN, model.OPN,
                         # function=lambda x: 0 if x >= 5 else 1)
        # nengo.Connection(model.SBG.LLBN, model.cbm, synapse=0.005)
        # nengo.Connection(model.OPN, model.SBG.EBN.neurons,
        #                  transform=inh_weight * np.ones((n_neurons, 1)))
        # nengo.Connection(model.OPN, model.SBG.IBN.neurons,
        #                  transform=inh_weight * np.ones((n_neurons, 1)))
        # nengo.Connection(model.OPN, model.SBG.LLBN.neurons,
        #                  transform=inh_weight * np.ones((n_neurons, 1)))
        nengo.Connection(model.SBG.AMN, model.plant[0])
        # nengo.Connection(model.SBG.OMN, model.plant[1])
        nengo.Connection(model.SBG.AMN, model.internal.control)
        # nengo.Connection(model.SBG.OMN, model.internal.control)
        # nengo.Connection(model.internal.pos, model.SBG.LLBN, transform=-1)
        nengo.Connection(model.internal.pos, model.SC_net.stim_transform[1])
        nengo.Connection(model.cbm.DCN, model.SBG.EBN, transform=200)

        model.epos = nengo.Node(size_in=1, label='eye_pos')
        model.vel = nengo.Node(size_in=1, label='eye_vel')
        model.force = nengo.Node(size_in=1, label='eye_force')
        nengo.Connection(model.plant[0], model.epos, synapse=None)
        nengo.Connection(model.plant[1], model.vel, synapse=None)
        nengo.Connection(model.plant[2], model.force, synapse=None)

        synapse = 0.005
        model.pulse_p = nengo.Probe(model.SBG.EBN_AMN, synapse=synapse)
        model.pos_p = nengo.Probe(model.epos, synapse=None)
        model.vel_p = nengo.Probe(model.vel, synapse=None)
        model.force_p = nengo.Probe(model.force, synapse=None)
        model.integ_p = nengo.Probe(model.SBG.TN_AMN, synapse=synapse)
        model.control_p = nengo.Probe(model.SBG.AMN, synapse=synapse)
        model.ipos_p = nengo.Probe(model.internal.pos, synapse=synapse)
        model.ivel_p = nengo.Probe(model.internal.vel, synapse=synapse)
        model.iforce_p = nengo.Probe(model.internal.force, synapse=synapse)
        model.OPN_p = nengo.Probe(model.OPN, synapse=synapse)
        model.IBN_p = nengo.Probe(model.SBG.IBN_OMN, synapse=synapse)
        model.LLBN_p = nengo.Probe(model.SBG.LLBN, synapse=synapse)
        model.SC_p = nengo.Probe(model.SC_net.output[0], synapse=synapse)
        model.DCN_p = nengo.Probe(model.cbm.DCN, synapse=synapse)
        model.IO_p = nengo.Probe(model.cbm.IO, synapse=synapse)
        model.PC_p = nengo.Probe(model.cbm.PCs, synapse=synapse)
        model.MF_p = nengo.Probe(model.cbm.MF, synapse=synapse)
        model.hpass_p = nengo.Probe(model.cbm.hpass_node, synapse=synapse)

    return model
