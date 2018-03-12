import numpy
np = numpy
import matplotlib.pyplot as plt

from control.simulation import Simulator, Experiment
from control.animation import Animator, Anim_2D
from control.environments.target import Target
from control.noise import Signal_Dep_Noise

from optimal_eye import Optimal_Eye
from eye_plant import Eye

from SBG_pair import run_SBG_pair
from internal_models import run_internal_model

############## Eye Animation #################################################

class Eye_Animation_1D(Anim_2D):

    def anim_update(self, data):
        self.line.set_data(data, 0.0)
        return [self.line]

############# set up trials ##################################################

dt = 0.001 # simulation timestep

if dt != 0.001:
    raise Warning("NEED TO CHANGE EYE MATRICES FOR NEW dt")

trial_length = 1
init_state = [[0], [0], [0], [0]]

k = 0.009 # signal-dep noise coefficient
noise = Signal_Dep_Noise([k])

############ set up objects ##################################################

# amplitudes are relative to initial position
#amplitudes = [5, 15, 25, 35, 45]
amplitudes = [90]
targets = [[amp, 0, 0, 0] for amp in amplitudes]

eye = Eye(dt=dt, init_state=init_state)
optimal = Optimal_Eye(eye)

env = Target(targets=targets, plant=eye, time_per_target=trial_length,
             dt=dt, x_left=-1, x_right=1, y_up=1, y_down=-1)

########### simulation functions #############################################

control_dim = 2

def optimal_init(sim):
    sim.plant.reset()
    target = sim.env.current_target
    sim.target = target[0, 0]
    control_seq = optimal.control_seq(init_state, sim.target, k)

    sim.control_seq = numpy.zeros((trial_length / dt, control_dim))
    duration = control_seq.shape[0] / control_dim
    sim.duration = duration
    sim.control_seq[0:duration, :] = control_seq.reshape((duration,
                                                          control_dim))
    sim.control = numpy.zeros((2, 1))

def step_func(sim, i, time):
    if i < sim.duration:
        sim.control = numpy.asmatrix(sim.control_seq[i]).T
        sim.control = noise(sim.control)
    else:
        sim.control = [[sim.target],  [0]]

    sim.plant.step(sim.control, dt)
    sim.env.step(dt)

########## simulate the model ################################################

sim = Simulator({'plant':eye, 'env':env},
                dt=dt, step_func=step_func, init_func=optimal_init)

pos = 'plant.meas[0, 0]'
vel = 'plant.meas[1, 0]'
force_ag = 'plant.meas[2, 0]'
force_ant = 'plant.meas[3, 0]'
# n_pos_ag = 'neural_pos_ag[i]'
n_pos_ag = 'n_pos_ag'
# n_pos_ant = 'neural_pos_ant[i]'
# n_vel = 'neural_data[1, i]'
# n_force = 'neural_data[2, i]'
control_ag = 'control[0]'
control_ant = 'control[1]'

sim.record(pos, vel, force_ag, force_ant, control_ag, control_ant)
           # n_pos_ag , n_pos_ant)

exper = Experiment(sim, trial_length, len(amplitudes))
exper.run()

################ animate #####################################################

# anim_objects = {'plant':Eye_Animation_1D(color='r', marker='o')}
# animator = Animator(title='Eye', xlim=45, ylim=2)
# animator.display_text(['plant'], -1, 1)
# animator.animate(data={'plant':exper.data(pos)},
#                  anim_objects=anim_objects,
#                  frames=exper.total_steps, ms_per_frame=5)

############# plot data ######################################################

def plot_experiments(amplitudes, exps, ylabel, function=None):

    for exp in exps:
        for i, amp in enumerate(amplitudes):
            if function is not None:
                data = function(exp, exper.data(exp, i))
            else:
                data = exper.data(exp, i)

            plt.plot(exper.trange, data, label='%s deg' % amp)

    # plt.legend(loc='best')
    plt.xlabel('time (s)')
    plt.ylabel(ylabel)

def helper(exp, x):
    if exp == n_pos_ag:
        return x * 2
    else:
        return x

plt.figure()
# plot_experiments(amplitudes, [vel, n_vel], 'vel', 'degress/s')
plt.subplot(231)
plot_experiments(amplitudes, [control_ag], 'control_ag')
plt.subplot(234)
plot_experiments(amplitudes, [control_ant], 'control_ant')
# plot_experiments(amplitudes, [pos], 'traj', 'degrees')
# plot_experiments(amplitudes, [pos, n_pos_ant], 'traj_with_neural_ant',
                 # 'degrees')
plt.subplot(232)
plot_experiments(amplitudes, [force_ag], 'force_ag')
plt.subplot(235)
plot_experiments(amplitudes, [force_ant], 'force_ant')
plt.subplot(233)
plot_experiments(amplitudes, [pos], 'true_pos (degrees)', function=helper)
plt.subplot(236)
plot_experiments(amplitudes, [vel], 'vel (degress/s)')
# plot_experiments(amplitudes, [force_ag, force_ant], 'force_ag,ant',
                 # 'force_ag,ant')
# plt.savefig("optimal")
plt.show()




##############################################################################
