import numpy as np
import matplotlib.pyplot as plt

import nengo

from eye_plant import Eye

dt = 0.001
stim_start = 0.1
trial_length = 1
pos = 0
pe = 0
K_p, K_d, K_i = 8, 0.1, 25
pulse_sat = 230

def make_network(target):

    init_state = [[0], [0], [0], [0]]
    l_eye = Eye(dt=dt, init_state=init_state)

    def l_plant_model(t, x):
        x = np.matrix(x).T
        l_eye.step(x, dt)
        if l_eye.state[2] > 250: l_eye.state[2] = 250
        return np.array(l_eye.state).squeeze()

    def cbm(t, ce):
        global pe
        temp = K_d * ((ce - pe) / dt)
        pe = ce
        return min(temp, pulse_sat)

    def integ(t, ce):
        global pos
        pos += ce * dt
        return K_i * pos

    def pulse(t, ce):
        if np.abs(ce) > 1:
            val = K_p * ce
            return min(val, pulse_sat)
        else:
            return 0

    with nengo.Network() as model:

        model.input_target = nengo.Node(output=lambda t: target \
                                        if t > stim_start else 0)
        model.error = nengo.Node(size_in=1, output=lambda t, x: x if x > 1 else 0)
        model.pulse = nengo.Node(size_in=1, output=pulse, label='Target_input')
        model.l_plant = nengo.Node(size_in=2, output=l_plant_model,
                                   label='l_eye')
        model.l_integ = nengo.Node(size_in=1, output=integ, label='Integ')
        model.cbm = nengo.Node(size_in=1, output=cbm, label='cbm')
        model.control = nengo.Node(size_in=1, label='Control')

        nengo.Connection(model.input_target, model.error)
        nengo.Connection(model.l_plant[0], model.error, transform=-1,
                         synapse=None)
        nengo.Connection(model.pulse, model.control, synapse=0)
        nengo.Connection(model.l_integ, model.control, synapse=0)
        nengo.Connection(model.cbm, model.control, synapse=0)
        nengo.Connection(model.control, model.l_plant[0], synapse=None)
        nengo.Connection(model.error, model.l_integ, synapse=None)
        nengo.Connection(model.error, model.pulse, synapse=None)
        nengo.Connection(model.error, model.cbm)

        model.l_pos = nengo.Node(size_in=1, label='l_eye_pos')
        model.l_vel = nengo.Node(size_in=1, label='l_eye_vel')
        model.l_force = nengo.Node(size_in=1, label='l_eye_force')
        nengo.Connection(model.l_plant[0], model.l_pos, synapse=None)
        nengo.Connection(model.l_plant[1], model.l_vel, synapse=None)
        nengo.Connection(model.l_plant[2], model.l_force, synapse=None)

    return model


def run_network(model, trial_length):

    with model:

        pulse = nengo.Probe(model.pulse, synapse=None)
        pos_lp = nengo.Probe(model.l_pos, synapse=None)
        vel_lp = nengo.Probe(model.l_vel, synapse=None)
        force_lp = nengo.Probe(model.l_force, synapse=None)
        integ_lp = nengo.Probe(model.l_integ, synapse=None)
        cbm_p = nengo.Probe(model.cbm, synapse=None)
        control_p = nengo.Probe(model.control, synapse=None)
        error_p = nengo.Probe(model.error, synapse=None)

    sim = nengo.Simulator(model, dt=dt)
    sim.run(trial_length)

    global pos, pe
    pos = 0
    pe = 0

    return {'pulse': sim.data[pulse], 'integ': sim.data[integ_lp],
            'pos':sim.data[pos_lp], 'vel':sim.data[vel_lp],
            'force':sim.data[force_lp], 't':sim.trange(),
            'cbm':sim.data[cbm_p], 'control':sim.data[control_p],
            'error':sim.data[error_p]}


def trials():

    targets = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45])
    # targets = np.array([20])
    peak_vel = []
    final_pos = []
    final_pos2 = []
    dur = []
    data = {}
    for target in targets:
        model = make_network(target=target)
        data[target] = run_network(model, trial_length=trial_length)
        peak_vel.append(np.amax(data[target]['vel']))
        for i, v in enumerate(data[target]['vel']):
            if i > (int(stim_start * 1000) + 30) and v < 30:
                dur.append(i / 1000. - (stim_start + 0.008))
                final_pos.append(data[target]['pos'][i])
                break
        final_pos2.append(data[target]['pos'][-1])


    t = data[targets[0]]['t']
    plt.figure(facecolor='white')
    axes = plt.axes(frameon=False)
    axes.set_frame_on(False)

    for target in targets:

        temp = data[target]
        plt.subplot(8, 1, 1)
        plt.plot(t, temp['error'])
        plt.ylabel('error')
        plt.subplot(8, 1, 2)
        plt.plot(t, temp['pulse'])
        plt.ylabel('pulse')
        plt.subplot(8, 1, 3)
        plt.plot(t, temp['integ'])
        plt.ylabel('integ')
        plt.subplot(8, 1, 4)
        plt.plot(t, temp['cbm'])
        plt.ylabel('cbm')
        plt.subplot(8, 1, 5)
        plt.plot(t, temp['control'])
        plt.ylabel('control')
        plt.subplot(8, 1, 6)
        plt.plot(t, temp['pos'])
        plt.ylabel('pos')
        plt.subplot(8, 1, 7)
        plt.plot(t, temp['vel'])
        plt.ylabel('vel')
        plt.subplot(8, 1, 8)
        plt.plot(t, temp['force'])
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
