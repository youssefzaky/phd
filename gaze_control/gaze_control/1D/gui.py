import nengo
import numpy as np

from complete import make_network, Target
from eye_plant import Eye


targets = np.array([20] * 30)
dt = 0.001
trial_length = 1
stim_start = 0.1
target_obj = Target(stim_start=stim_start)
init_state = [[0], [0], [0], [0]]
eye = Eye(dt=dt, init_state=init_state)
model = make_network(eye, target_obj, dt)
#sim = nengo.Simulator(model, dt=dt)

#for target in targets:
#    target_obj.target = target
#    sim.reset()
#    eye.reset()