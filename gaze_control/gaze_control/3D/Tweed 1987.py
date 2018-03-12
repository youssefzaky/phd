import numpy as np
import matplotlib.pyplot as plt
import nengo

axis = [0, 0, 1]
angle = np.pi
sine = np.sin(angle/2)
q_d = np.array([np.cos(angle/2), sine*axis[0], sine*axis[1], sine*axis[2]])
q_d /= np.linalg.norm(q_d)
print q_d

def quat_inv(q):
    return [q[0], -q[1], -q[2], -q[3]]

def multiply(q1, q2):
    r1, r2, v1, v2 = q1[0], q2[0], np.array(q1[1:]), np.array(q2[1:])
    v = r1 * v2 + r2*v1 + np.cross(v1, v2)
    return r1*r2 - np.dot(v1, v2), v[0], v[1], v[2]

def multiply2(x):
    q1, q2 = x[:4], x[4:]
    return multiply(q1, q2)

def derivative(x):
    q1, q2 = x[1:4], x[4:]
    q1 = [0, q1[0], q1[1], q1[2]]
    return multiply(q1, q2)

rec_tau = 0.1
start_time = 0.1

from nengo.dists import UniformHypersphere as UH

eval_points = np.hstack((UH(surface=True).sample(5000, 4),  UH(surface=True).sample(5000, 4)))


with nengo.Network() as model:
    # model.config[nengo.Ensemble].neuron_type = nengo.Direct()
    q_id = nengo.Node(output=lambda t: [1, 0, 0, 0] if t < start_time else [0, 0, 0, 0])
    quat_in = nengo.Node(output=lambda t: [0, 0, 0, 0] if t < start_time else q_d) # input
    target = nengo.Ensemble(dimensions=4, n_neurons=1000, eval_points=UH(True)) # desired orientation
    q = nengo.Ensemble(dimensions=4, n_neurons=3000, eval_points=UH(True)) # orientation of eye
    pre_error = nengo.Ensemble(dimensions=8, n_neurons=3000, radius=1.5) # pre multiplication
    error = nengo.Ensemble(dimensions=4, n_neurons=1000, eval_points=UH(True)) # rotation error
    pre_int = nengo.Ensemble(dimensions=8, n_neurons=3000, radius=1.5)
    nengo.Connection(quat_in, target)
    nengo.Connection(q_id, q)
    nengo.Connection(target, pre_error[:4])
    nengo.Connection(q, pre_error[4:], function=quat_inv) # invert current orientation
    nengo.Connection(pre_error, error, function=multiply2) # quaternion multiplication
    nengo.Connection(error[1:], pre_int[1:4], transform=.5)
    nengo.Connection(q, pre_int[4:])
    nengo.Connection(pre_int, q, transform=rec_tau, function=derivative)
    nengo.Connection(q, q, synapse=rec_tau)
    p = nengo.Probe(q, synapse=0.1)

# import nengo_gui
# nengo_gui.GUI(__file__).start()

sim = nengo.Simulator(model)
sim.run(15)

plt.plot(sim.trange(), np.ones(len(sim.trange()))*q_d[0], 'r--', lw=2)
plt.plot(sim.trange(), np.ones(len(sim.trange()))*q_d[1], 'g--', lw=2)
plt.plot(sim.trange(), np.ones(len(sim.trange()))*q_d[2], 'b--', lw=2)
plt.plot(sim.trange(), np.ones(len(sim.trange()))*q_d[3], 'y--', lw=2)

plt.plot(sim.trange(), sim.data[p][:, 1], label='torsion')
plt.plot(sim.trange(), sim.data[p][:, 0])
plt.plot(sim.trange(), sim.data[p][:, 2])
plt.plot(sim.trange(), sim.data[p][:, 3])
plt.ylim(-1, 1.2)
plt.legend(loc='best')
plt.show()
