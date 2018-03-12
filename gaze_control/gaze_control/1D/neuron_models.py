import matplotlib.pyplot as plt
import nengo
from nengo.utils.neurons import rates_isi
from nengo.utils.ensemble import tuning_curves

dur = 0.1  # duration of stimulus
dt = 0.0001  # use finer resolution cz of high bursting

with nengo.Network() as model:
    # parameters tuned for EBN/IBN
    neuron = nengo.Izhikevich(reset_voltage=-70, reset_recovery=2,
                              tau_recovery=0.005)

    ens = nengo.Ensemble(n_neurons=1, dimensions=1, encoders=[[1]],
                         intercepts=[-0.9], max_rates=[150],
                         neuron_type=neuron)
    # ens = nengo.Ensemble(n_neurons=1, dimensions=1, encoders=[[1]],
    # intercepts=[-0.9])
    node = nengo.Node(output=lambda t: 1 if t < dur else -1)
    nengo.Connection(node, ens)
    sp = nengo.Probe(ens.neurons, attr='spikes')
    cp = nengo.Probe(ens, attr='input')

s = nengo.Simulator(model, dt=dt)
s.run(0.1)

plt.figure()
plt.subplot(221)
plt.plot(s.trange(), s.data[sp])
plt.subplot(222)
plt.plot(s.trange(), rates_isi(s.trange(), s.data[sp]))
plt.subplot(223)
plt.plot(s.trange(), s.data[cp])
plt.subplot(224)
plt.plot(*tuning_curves(ens, s))
plt.show()
