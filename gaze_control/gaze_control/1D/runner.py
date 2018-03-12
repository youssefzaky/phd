import matplotlib.pyplot as plt
import numpy as np
import nengo

from complete import make_network, Target
from eye_plant import Eye

def trials():

    # targets = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45])
    targets = np.array([20] * 30)
    # targets = np.array([20])
    peak_vel = []
    final_pos = []
    final_pos2 = []
    dur = []
    data = {}
    dt = 0.001
    trial_length = 0.35
    stim_start = 0.05
    stim_end = 0.15

    plt.figure(facecolor='white')
    axes = plt.axes(frameon=False)
    axes.set_frame_on(False)
    target_obj = Target(stim_start=stim_start, stim_end=stim_end)
    init_state = [[0], [0], [0], [0]]
    eye = Eye(dt=dt, init_state=init_state)
    model = make_network(eye, target_obj, dt)
    sim = nengo.Simulator(model, dt=dt)

    for j, target in enumerate(targets):
        target_obj.target = target
        sim.reset()
        eye.reset()
        sim.run(trial_length)
        data[j] = sim.data

        peak_vel.append(np.amax(data[j][model.vel_p]))
        for i, v in enumerate(data[j][model.vel_p]):
            if i > (int(stim_start * 1000) + 30) and v < 30:
                dur.append(i / 1000. - (stim_start + 0.008))
                final_pos.append(data[j][model.pos_p][i])
                break
        final_pos2.append(data[j][model.pos_p][-1])

        t = sim.trange()

        temp = data[j]
        plt.subplot(14, 1, 1)
        plt.plot(t, temp[model.SC_p])
        plt.ylabel('SC')
        plt.subplot(14, 1, 2)
        plt.plot(t, temp[model.LLBN_p])
        plt.ylabel('LLBN')
        plt.subplot(14, 1, 3)
        plt.plot(t, temp[model.OPN_p])
        plt.ylabel('OPN')
        plt.subplot(14, 1, 4)
        plt.plot(t, temp[model.pulse_p])
        plt.ylabel('pulse')
        plt.subplot(14, 1, 5)
        plt.plot(t, temp[model.integ_p])
        plt.ylabel('integ')
        plt.subplot(14, 1, 6)
        plt.plot(t, temp[model.hpass_p])
        plt.ylabel('hpass')
        plt.subplot(14, 1, 7)
        plt.plot(t, temp[model.PC_p])
        plt.ylabel('PC')
        plt.subplot(14, 1, 8)
        plt.plot(t, temp[model.IO_p])
        plt.ylabel('IO')
        plt.subplot(14, 1, 9)
        plt.plot(t, temp[model.DCN_p])
        plt.ylabel('DCN')
        plt.subplot(14, 1, 10)
        plt.plot(t, temp[model.IBN_p])
        plt.ylabel('IBN')
        plt.subplot(14, 1, 11)
        plt.plot(t, temp[model.control_p])
        plt.ylabel('control')
        plt.subplot(14, 1, 12)
        plt.plot(t, temp[model.pos_p])
        # plt.plot(t, temp[model.ipos_p])
        plt.ylabel('pos')
        plt.subplot(14, 1, 13)
        plt.plot(t, temp[model.vel_p])
        # plt.plot(t, temp[model.ivel_p])
        plt.ylabel('vel')
        plt.subplot(14, 1, 14)
        plt.plot(t, temp[model.force_p])
        # plt.plot(t, temp[model.iforce_p])
        plt.ylabel('force')
        # plt.show()

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    # plt.figure()
    # plt.plot(targets, peak_vel, label='peak-vel vs amp')
    # plt.legend(loc='best')
    # plt.figure()
    # plt.plot(targets, final_pos, label='1 pos accuracy')
    # plt.plot(targets, final_pos2, label='2 pos accuracy')
    # plt.plot(targets, targets, label='truth')
    # plt.legend(loc='best')
    # plt.figure()
    # plt.plot(targets, dur, label='dur vs amp')
    # plt.legend(loc='best')
    plt.show()

trials()
