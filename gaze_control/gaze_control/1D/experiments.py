from __future__ import division
import nengo
import numpy 
import matplotlib.pyplot as plt
from Controller_2D import Controller_2D

rng = numpy.random.RandomState()

def amplitude_var(samples=500):
    angles = rng.uniform(0.1, 0.9, samples)
    magnitudes = rng.uniform(0.2, 0.9, samples)
    return angles, magnitudes

def saccade_dur(array, start_time, dt):
    index = numpy.argmax(array)
    return index * dt - start_time

filter = 0.07

model = nengo.Model("Eye Control")

hor_val, vert_val = 0, 0

def sacc(t):
    return [hor_val * (t > 0.1) * (t < 0.18), vert_val * (t > 0.1) * (t < 0.18)] 

with model:
        
    control = Controller_2D(exc=0.007, inh=0.05, inh_weight=-0.5, OPN_weight=-0.001, hgain=3 , vgain=3)

    MN_hr_p = nengo.Probe(control.hor.right.MN, filter=filter)
    MN_vr_p = nengo.Probe(control.ver.right.MN, filter=filter)
    sac_input = nengo.Node(output=sacc, label="Saccade Vector")
    
    #connect to left SC
    nengo.Connection(sac_input, control.SC_left, filter=None)

def run_experiment(sacc, t):    

    sim = nengo.Simulator(model, dt = 0.001, builder=nengo.builder.Builder(copy=False))
    sim.run(t)

    return sim.data(MN_hr_p), sim.data(MN_vr_p)

def amp_vs_dur():
    global hor_val, vert_val

    samples = 10
    trials = 5
    x, y, a = amplitude_var(samples=samples)
    d = numpy.zeros(samples)
    amp = numpy.zeros(samples)


    for i in range(samples):

        print 'input: ', x[i], y[i]

        hor_val = x[i]
        vert_val = y[i]

        print "percentage completed: ", i / samples

        for j in range(trials):
            h, v = run_experiment(sacc, 0.4)
            hd = saccade_dur(h[:,0], 0.13, 0.001)
            amp[i] += numpy.amax(h[:,0])
            d[i] += hd


    d /= trials
    amp /= trials

    plt.figure()
    plt.plot(amp, d, 'x')
    plt.savefig("amp. vs dur.")

def amp_vs_peak_vel():
    global hor_val, vert_val

    samples = 10
    trials = 5
    x, y, a = amplitude_var(samples=samples)
    d = numpy.zeros(samples)


    for i in range(samples):

        hor_val = x[i]
        vert_val = y[i]

        print x[i]

        print "percentage completed: ", (i + 1) / samples

        for j in range(trials):
            h, v = run_experiment(sacc, 0.4)
            hd = numpy.amax(h[:,1])
            vd = numpy.amax(v[:,1])
            d[i] += hd

    d /= trials

    plt.figure()
    plt.plot(a, d, 'x')
    plt.savefig("amp. vs peak_vel.")

def skewness():
    global hor_val, vert_val

    samples = 1
    trials = 5
    x, y, a = amplitude_var(samples=samples)
    d = numpy.zeros(1000)

    for i in range(samples):

        print x[i]
        hor_val = x[i]
        vert_val = y[i]

        print "percentage completed: ", (i + 1) / samples
        for j in range(trials):
            h, v = run_experiment(sacc, 1)
            d += h[:,1]

    d /= trials
    
    plt.figure()
    plt.plot(numpy.arange(1000), d, 'x')
    plt.savefig("skewness")

def straight():
    global hor_val, vert_val

    samples = 15
    trials = 1
    angles, magnitudes = amplitude_var(samples=samples)
    hor = numpy.zeros((200, samples))
    ver = numpy.zeros((200, samples))

    for i in range(samples):

        hor_val = magnitudes[i]
        vert_val = angles[i]

        print "percentage completed: ", (i + 1) / samples

        for j in range(trials):
            h, v = run_experiment(sacc, 0.20)
            hor[:,i] += h[:,0]
            ver[:,i] += v[:,0]

    hor /= trials
    ver /= trials

    plt.figure()

    for i in range(samples):
        plt.plot(hor[:,i], ver[:,i])

    plt.savefig("straight_lines")


skewness()
#amp_vs_peak_vel()
#amp_vs_dur()
#straight()
