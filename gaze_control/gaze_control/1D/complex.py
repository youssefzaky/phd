import numpy as np
import matplotlib.pyplot as plt

import nengo

from eye_plant import Eye
from internal_models import Internal_Model

######### Note: check tuning curves #############

stim_start = 0.1
pe = 0

def make_network(target, dt):

    init_state = [[0], [0], [0], [0]]
    l_eye = Eye(dt=dt, init_state=init_state)
    K_p, K_d, K_i = 8, 0.1, 30
    pulse_sat = 200
    n_neurons = 300
    pos_encs = np.ones((n_neurons, 1))
    rec_tau = 0.005
    con_radius = 150
    int_radius = 45
    inh_weight = -0.001
    intercepts = np.random.uniform(0, 0.9, size=n_neurons)

    def plant_model(t, x):
        x = np.matrix(x).T
        l_eye.step(x, dt)
        if l_eye.state[2] > 250: l_eye.state[2] = 250
        return np.array(l_eye.state).squeeze()

    def cbm(t, ce):
        global pe
        temp = K_d * ((ce - pe) / dt)
        pe = ce
        return min(temp, 100)

    with nengo.Network() as model:

        model.target = nengo.Node(output=lambda t: target if t > stim_start
                                  else 0)
        model.plant = nengo.Node(size_in=2, output=plant_model,
                                   label='l_eye')
        model.cbm = nengo.Node(size_in=1, output=cbm, label='cbm')

        model.OPN = nengo.Ensemble(dimensions=1, n_neurons=n_neurons,
                                   encoders=pos_encs, intercepts=intercepts)
        model.SC = nengo.Ensemble(dimensions=1, n_neurons=n_neurons,
                                  radius=45, label='SC', intercepts=intercepts,
                                  encoders=pos_encs)
        model.LLBN = nengo.Ensemble(dimensions=1, n_neurons=n_neurons,
                                    radius=45, encoders=pos_encs,
                                    intercepts=intercepts, label='LLBN')
        model.pulse = nengo.Ensemble(dimensions=1, n_neurons=n_neurons,
                                     radius=pulse_sat, encoders=pos_encs,
                                     intercepts=intercepts, label='EBN')
        model.IBN = nengo.Ensemble(dimensions=1, n_neurons=n_neurons,
                                     radius=pulse_sat, encoders=-pos_encs,
                                     intercepts=intercepts, label='EBN')
        model.integ = nengo.Ensemble(n_neurons=1000, dimensions=1,
                                       radius=int_radius, label='BTN')
        model.control = nengo.Ensemble(n_neurons=n_neurons, dimensions=1,
                                       encoders=pos_encs, intercepts=intercepts,
                                       radius=con_radius, label='AMN')
        model.internal = Internal_Model(l_eye, neurons=1000)

        nengo.Connection(model.target, model.SC, synapse=None)
        nengo.Connection(model.SC, model.LLBN)
        nengo.Connection(model.LLBN, model.OPN,
                         function=lambda x: 0 if x >= 5 else 1)
        nengo.Connection(model.LLBN, model.integ, transform=K_i * rec_tau)
        nengo.Connection(model.LLBN, model.pulse, transform=K_p)
        nengo.Connection(model.LLBN, model.cbm, synapse=0.005)
        nengo.Connection(model.OPN, model.pulse.neurons,
                         transform=inh_weight * np.ones((n_neurons, 1)))
        nengo.Connection(model.OPN, model.IBN.neurons,
                         transform=inh_weight * np.ones((n_neurons, 1)))
        nengo.Connection(model.OPN, model.LLBN.neurons,
                         transform=inh_weight * np.ones((n_neurons, 1)))
        model.pulse_out = nengo.Connection(model.pulse, model.control)
        nengo.Connection(model.integ, model.control)
        nengo.Connection(model.integ, model.integ, synapse=rec_tau)
        nengo.Connection(model.cbm, model.control)
        nengo.Connection(model.control, model.plant[0])
        nengo.Connection(model.control, model.internal.control)
        nengo.Connection(model.internal.pos, model.LLBN, transform=-1)

        model.pos = nengo.Node(size_in=1, label='eye_pos')
        model.vel = nengo.Node(size_in=1, label='eye_vel')
        model.force = nengo.Node(size_in=1, label='eye_force')
        nengo.Connection(model.plant[0], model.pos, synapse=None)
        nengo.Connection(model.plant[1], model.vel, synapse=None)
        nengo.Connection(model.plant[2], model.force, synapse=None)

        synapse = 0.005
        model.pulse_p = nengo.Probe(model.pulse_out, synapse=synapse)
        model.pos_p = nengo.Probe(model.pos, synapse=None)
        model.vel_p = nengo.Probe(model.vel, synapse=None)
        model.force_p = nengo.Probe(model.force, synapse=None)
        model.integ_p = nengo.Probe(model.integ, synapse=synapse)
        model.cbm_p = nengo.Probe(model.cbm, synapse=synapse)
        model.control_p = nengo.Probe(model.control, synapse=synapse)
        model.ipos_p = nengo.Probe(model.internal.pos, synapse=synapse)
        model.ivel_p = nengo.Probe(model.internal.vel, synapse=synapse)
        model.iforce_p = nengo.Probe(model.internal.force, synapse=synapse)
        model.OPN_p = nengo.Probe(model.OPN, synapse=synapse)
        model.IBN_p = nengo.Probe(model.IBN, synapse=synapse)
        model.LLBN_p = nengo.Probe(model.LLBN, synapse=synapse)
        model.SC_p = nengo.Probe(model.SC, synapse=synapse)

    return model


def trials():

    targets = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45])
    # targets = np.array([20, 30])
    peak_vel = []
    final_pos = []
    final_pos2 = []
    dur = []
    data = {}
    dt = 0.001
    trial_length = 1

    plt.figure(facecolor='white')
    axes = plt.axes(frameon=False)
    axes.set_frame_on(False)

    for target in targets:
        model = make_network(target, dt)
        sim = nengo.Simulator(model, dt=dt)
        sim.run(trial_length)
        global pe
        pe = 0
        data[target] = sim.data

        peak_vel.append(np.amax(data[target][model.vel_p]))
        for i, v in enumerate(data[target][model.vel_p]):
            if i > (int(stim_start * 1000) + 30) and v < 30:
                dur.append(i / 1000. - (stim_start + 0.008))
                final_pos.append(data[target][model.pos_p][i])
                break
        final_pos2.append(data[target][model.pos_p][-1])

        t = sim.trange()

        temp = data[target]
        plt.subplot(11, 1, 1)
        plt.plot(t, temp[model.SC_p])
        plt.ylabel('SC')
        plt.subplot(11, 1, 2)
        plt.plot(t, temp[model.LLBN_p])
        plt.ylabel('LLBN')
        plt.subplot(11, 1, 3)
        plt.plot(t, temp[model.OPN_p])
        plt.ylabel('OPN')
        plt.subplot(11, 1, 4)
        plt.plot(t, temp[model.pulse_p])
        plt.ylabel('pulse')
        plt.subplot(11, 1, 5)
        plt.plot(t, temp[model.integ_p])
        plt.ylabel('integ')
        plt.subplot(11, 1, 6)
        plt.plot(t, temp[model.cbm_p])
        plt.ylabel('deriv')
        plt.subplot(11, 1, 7)
        plt.plot(t, temp[model.IBN_p])
        plt.ylabel('IBN')
        plt.subplot(11, 1, 8)
        plt.plot(t, temp[model.control_p])
        plt.ylabel('control')
        plt.subplot(11, 1, 9)
        plt.plot(t, temp[model.pos_p])
        # plt.plot(t, temp[model.ipos_p])
        plt.ylabel('pos')
        plt.subplot(11, 1, 10)
        plt.plot(t, temp[model.vel_p])
        # plt.plot(t, temp[model.ivel_p])
        plt.ylabel('vel')
        plt.subplot(11, 1, 11)
        plt.plot(t, temp[model.force_p])
        # plt.plot(t, temp[model.iforce_p])
        plt.ylabel('force')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.figure()
    plt.plot(targets, peak_vel, label='peak-vel vs amp')
    plt.legend(loc='best')
    plt.figure()
    plt.plot(targets, final_pos, label='1 pos accuracy')
    plt.plot(targets, final_pos2, label='2 pos accuracy')
    plt.plot(targets, targets, label='truth')
    plt.legend(loc='best')
    plt.figure()
    plt.plot(targets, dur, label='dur vs amp')
    plt.legend(loc='best')
    plt.show()

trials()
