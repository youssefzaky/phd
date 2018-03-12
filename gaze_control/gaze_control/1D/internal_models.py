import numpy as np
import nengo

import sys
sys.path.append("/home/youssef/GitHub/cerebellum/")

# from Cerebellum import Microzone


############### internal neural model ########################################


class Internal_Model(nengo.Network):

    def __init__(self, eye, neurons=500, pos_radius=45, vel_radius=900,
                 f_radius=250, integ_tau=0.008, **ens_kwargs):

        self.pos = nengo.Ensemble(n_neurons=neurons, dimensions=1,
                                  radius=pos_radius, label='pos', **ens_kwargs)
        self.vel = nengo.Ensemble(n_neurons=neurons, dimensions=1,
                                  radius=vel_radius, label='vel', **ens_kwargs)
        self.force = nengo.Ensemble(n_neurons=neurons, dimensions=1,
                                    radius=f_radius, label='force', **ens_kwargs)
        self.control = nengo.Node(size_in=1)

        tau = integ_tau # to save typing
        self.tau = tau # for outside reference
        self.eye = eye

        # recurrent connections
        nengo.Connection(self.pos, self.pos, synapse=tau)
        nengo.Connection(self.vel, self.vel,
                         transform=[[(tau * -eye.b / eye.m) + 1]],
                         synapse=tau)
        nengo.Connection(self.force, self.force,
                         transform=[[tau * -eye.alpha_2 / eye.alpha_1 + 1]],
                         synapse=tau)

        # mutual connections
        nengo.Connection(self.pos, self.vel, transform=[[tau * -eye.k / eye.m]])
        nengo.Connection(self.force, self.vel, transform=[[tau * 1 / eye.m]])
        nengo.Connection(self.vel, self.pos, transform=[[tau]])
        nengo.Connection(self.control, self.force, transform=[[tau / eye.alpha_1]])


def run_internal_model(eye, control, dt, trial_length):
    # note: dt must be the same as the non-neural simulation dt

    control_ag = control[:, 0]
    control_ant = control[:, 1]

    def input_ag(t):
        return control_ag[np.minimum(t / dt, trial_length / dt - 1)]

    def input_ant(t):
        return control_ant[np.minimum(t / dt, trial_length / dt - 1)]

    with nengo.Network() as model:

        activation_ag = nengo.Node(output=input_ag)
        activation_ant = nengo.Node(output=input_ant)

        internal_ag = Internal_Model(eye)
        internal_ant = Internal_Model(eye)

        tau = internal_ag.tau

        # inputs
        nengo.Connection(activation_ag, internal_ag.force,
                         transform=[[tau * 1 / eye.alpha_1]],
                         synapse=None)

        nengo.Connection(activation_ant, internal_ant.force,
                         transform=[[tau * 1 / eye.alpha_1]],
                         synapse=None)

        # force cross-connection
        nengo.Connection(internal_ag.force, internal_ant.vel,
                         transform=[[tau * - 1 / eye.m]])
        nengo.Connection(internal_ant.force, internal_ag.vel,
                         transform=[[tau * - 1 / eye.m]])

    probe_dict = {'p_ag': internal_ag.p_probe, 'v_ag': internal_ag.v_probe,
                  'f_ag': internal_ag.f_probe, 'p_ant': internal_ant.p_probe,
                  'v_ant': internal_ant.v_probe, 'f_ant': internal_ant.f_probe}

    sim = nengo.Simulator(model, dt=dt)
    sim.run(trial_length)
    return sim.data, probe_dict


################ internal CB model ############################################


def cb_model(eye, control, dt, trial_length):

    rec_tau = 0.1

    def node_out(t):
        return control[(t / dt) - 1, 0]

    def rec_func(x):
        control = x[-1]
        state = x[:3]
        f = state + rec_tau * np.dot(eye.c_A, state)
        g = (rec_tau * np.dot(eye.c_B, control)).flatten()
        return np.hstack((f + g, 0))

    # note: dt must be the same as the non-neural simulation dt
    with nengo.Network() as model:

        activation = nengo.Node(output=node_out, label='act')
        mz = Microzone(rec_func, PC_n=300, GC_n=300, DCN_n=10,
                       IO_n=10, MF_dim=4, IO_dim=3,
                       radius_input=np.sqrt(3))

        # input
        nengo.Connection(activation, mz.MF_input[-1], synapse=None)

        p_probe = nengo.Probe(mz.PC[0], synapse=probe_filter)
        v_probe = nengo.Probe(mz.PC[1], synapse=probe_filter)
        f_probe = nengo.Probe(mz.PC[2], synapse=probe_filter)

    sim = nengo.Simulator(model, dt=dt)
    sim.run(trial_length)
    return np.array([sim.data[p_probe], sim.data[v_probe], sim.data[f_probe]])
