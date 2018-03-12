import numpy as np
import matplotlib.pyplot as plt

import nengo
from nengo.dists import Uniform

from SBG_pair import SBG_Pair
from eye_plant import Eye

from control.noise import Signal_Dep_Noise

k = 0.009 # signal-dep noise coefficient
noise = Signal_Dep_Noise([k])

dt = 0.001
init_state = [[0], [0], [0], [0]]
r_eye = Eye(dt=dt, init_state=init_state)
l_eye = Eye(dt=dt, init_state=init_state)

r_eye.reset()
l_eye.reset()


def r_plant_model(t, x):
    x = np.matrix(x).T
    x = noise(x)
    r_eye.step(x, dt)
    return np.array(r_eye.state).squeeze()


def l_plant_model(t, x):
    x = np.matrix(x).T
    x = noise(x)
    l_eye.step(x, dt)
    return np.array(l_eye.state).squeeze()


amplitude = 10
target = amplitude / 45.
duration = (2.7 * amplitude) / 1000 # duration in ms
stim_start = 0.1
end_time = stim_start + duration

EBN_gain, TN_gain, MN_gain = 2.8, 1, 80

def make_network(target):

    with nengo.Network() as model:

        #model.config[nengo.Ensemble].neuron_type = nengo.Direct()

        # networks
        # model.SC_l = nengo.Ensemble(dimensions=1, n_neurons=200, radius=1,
        #                     encoders=Uniform(1, 1), intercepts=Uniform(0, 0.1),
        #                     label='SC_l')
        model.SC_l = nengo.Ensemble(n_neurons=150, dimensions=2, encoders=Uniform(1, 1))

        model.SBG = SBG_Pair(label='SBG', include_OPN=False, EBN_gain=EBN_gain,
                        TN_gain=TN_gain, rec_tau=0.1)

        # nodes
        model.sacc_input = nengo.Node(output=lambda t: [target] if t > 0.1 and
                                      t < 0.13 else 0, label='Target_input')
        model.r_plant = nengo.Node(size_in=2, output=r_plant_model,
                                   label='r_eye')
        model.l_plant = nengo.Node(size_in=2, output=r_plant_model,
                                   label='l_eye')

        alpha = 25.
        beta = alpha / 4.

        # target to SC
        # nengo.Connection(model.sacc_input, model.SC_l, synapse=None)
        nengo.Connection(model.SC_l, model.SC_l, transform=np.eye(2) \
                         + np.array([[0, 1], [-alpha*beta, -alpha]]),
                         synapse=0.1)
        # send goal input in to ddyz
        nengo.Connection(model.sacc_input, model.SC_l[1],
                         transform=[[alpha*beta]],
                         synapse=None)

        # SC to SBG
        nengo.Connection(model.SC_l[1], model.SBG.right.LLBN)
        #nengo.Connection(model.SC, model.SBG.right.EBN)

        # SBG to plant node

        nengo.Connection(model.SBG.right.AMN, model.r_plant[0],
                         transform=MN_gain)
        nengo.Connection(model.SBG.right.OMN, model.r_plant[1],
                         transform=MN_gain)

        # SBG to plant node
        nengo.Connection(model.SBG.left.OMN, model.l_plant[0],
                         transform=MN_gain, synapse=None)
        nengo.Connection(model.SBG.left.AMN, model.l_plant[1],
                         transform=MN_gain, synapse=None)

        #nengo.Connection(SC, SBG.OPN, function= lambda x: 90 if x < 5 else 0)

        model.r_pos = nengo.Node(size_in=1, label='r_eye_pos')
        model.r_vel = nengo.Node(size_in=1, label='r_eye_vel')
        model.r_force = nengo.Node(size_in=1, label='r_eye_force')
        nengo.Connection(model.r_plant[0], model.r_pos, synapse=None)
        nengo.Connection(model.r_plant[1], model.r_vel, synapse=None)
        nengo.Connection(model.r_plant[2], model.r_force, synapse=None)

        model.l_pos = nengo.Node(size_in=1, label='l_eye_pos')
        model.l_vel = nengo.Node(size_in=1, label='l_eye_vel')
        model.l_force = nengo.Node(size_in=1, label='l_eye_force')
        nengo.Connection(model.l_plant[0], model.l_pos, synapse=None)
        nengo.Connection(model.l_plant[1], model.l_vel, synapse=None)
        nengo.Connection(model.l_plant[2], model.l_force, synapse=None)

        #nengo.Connection(r_pos, SC, transform=-1)

        model.stop = nengo.Node(output=lambda t: 2 if t > end_time and \
                                t < end_time + 0.005 else 0, label='CB')
        # nengo.Connection(model.stop, model.SBG.left.EBN)
        # nengo.Connection(model.stop, model.SBG.left.IBN)

    return model


model = make_network(target=target)

def run_network(model, trial_length, with_plots=True):

    with model:

        circuit = model.SBG

        filter = 0.005

        input_p = nengo.Probe(model.sacc_input, synapse=None)

        # left probes
        SC_lp = nengo.Probe(model.SC_l, synapse=filter)
        LLBN_lp = nengo.Probe(circuit.left.LLBN, synapse=filter)
        EBN_lp = nengo.Probe(circuit.left.EBN, synapse=filter)
        TN_lp = nengo.Probe(circuit.left.TN, synapse=filter)
        IBN_lp = nengo.Probe(circuit.left.IBN, synapse=filter)
        OMN_lp = nengo.Probe(circuit.left.OMN, synapse=filter)
        AMN_lp = nengo.Probe(circuit.left.AMN, synapse=filter)
        pos_rp = nengo.Probe(model.r_pos, synapse=None)
        vel_rp = nengo.Probe(model.r_vel, synapse=None)
        force_rp = nengo.Probe(model.r_force, synapse=None)

        # right probes
        LLBN_rp = nengo.Probe(circuit.right.LLBN, synapse=filter)
        EBN_rp = nengo.Probe(circuit.right.EBN, synapse=filter)
        TN_rp = nengo.Probe(circuit.right.TN, synapse=filter)
        IBN_rp = nengo.Probe(circuit.right.IBN, synapse=filter)
        OMN_rp = nengo.Probe(circuit.right.OMN, synapse=filter)
        AMN_rp = nengo.Probe(circuit.right.AMN, synapse=filter)
        pos_lp = nengo.Probe(model.l_pos, synapse=None)
        vel_lp = nengo.Probe(model.l_vel, synapse=None)
        force_lp = nengo.Probe(model.l_force, synapse=None)


    sim = nengo.Simulator(model, dt=dt)
    sim.run(trial_length)
    # reset velocities if you want to do multiple saccades

    if with_plots:

        t = sim.trange()
        plt.figure(facecolor='white')
        axes = plt.axes(frameon=False)
        axes.set_frame_on(False)

        # plot left
        ax1 = plt.subplot(11, 2, 1)
        plt.plot(t, sim.data[SC_lp], label='SC_l')
        ax1.set_ylim(0, 1)
        # plt.legend(loc='best')
        plt.subplot(11, 2, 4, sharey=ax1)
        plt.plot(t, sim.data[LLBN_lp], label='LLBN-l')
        plt.legend(loc='best')
        plt.subplot(11, 2, 6, sharey=ax1)
        plt.plot(t, sim.data[EBN_lp], label='EBN-l')
        plt.legend(loc='best')
        plt.subplot(11, 2, 8, sharey=ax1)
        plt.plot(t, sim.data[IBN_lp], label='IBN-l')
        plt.legend(loc='best')
        plt.subplot(11, 2, 10, sharey=ax1)
        plt.plot(t, sim.data[TN_lp], label='TN-l')
        plt.legend(loc='best')
        plt.subplot(11, 2, 12, sharey=ax1)
        plt.plot(t, sim.data[OMN_lp], label='OMN-l')
        plt.legend(loc='best')
        plt.subplot(11, 2, 14, sharey=ax1)
        plt.plot(t, sim.data[AMN_lp], label='AMN-l')
        plt.legend(loc='best')
        plt.subplot(11, 2, 16)
        plt.plot(t, sim.data[pos_lp], label='pos-l')
        plt.legend(loc='best')
        plt.subplot(11, 2, 18)
        plt.plot(t, sim.data[vel_lp], label='vel-l')
        plt.legend(loc='best')
        plt.subplot(11, 2, 20)
        plt.plot(t, sim.data[force_lp], label='force-l')
        plt.legend(loc='best')

        # plot right
        plt.subplot(11, 2, 2)
        plt.plot(t, sim.data[input_p], label='Input-l')
        plt.legend(loc='best')
        plt.subplot(11, 2, 3)
        plt.plot(t, sim.data[LLBN_rp], label='LLBN-r')
        plt.legend(loc='best')
        plt.subplot(11, 2, 5)
        plt.plot(t, sim.data[EBN_rp], label='EBN-r')
        plt.legend(loc='best')
        plt.subplot(11, 2, 7)
        plt.plot(t, sim.data[IBN_rp], label='IBN-r')
        plt.legend(loc='best')
        plt.subplot(11, 2, 9)
        plt.plot(t, sim.data[TN_rp], label='TN-r')
        plt.legend(loc='best')
        plt.subplot(11, 2, 11)
        plt.plot(t, sim.data[OMN_rp], label='OMN-r')
        plt.legend(loc='best')
        plt.subplot(11, 2, 13)
        plt.plot(t, sim.data[AMN_rp], label='AMN-r')
        plt.legend(loc='best')
        plt.subplot(11, 2, 15)
        plt.plot(t, sim.data[pos_rp], label='pos-r')
        plt.legend(loc='best')
        plt.subplot(11, 2, 17)
        plt.plot(t, sim.data[vel_rp], label='vel-r')
        plt.legend(loc='best')
        plt.subplot(11, 2, 19)
        plt.plot(t, sim.data[force_rp], label='force-r')
        plt.legend(loc='best')


        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

run_network(model, trial_length=0.3)
