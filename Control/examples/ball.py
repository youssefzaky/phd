import numpy

from control.plants.object_2D import Object_2D
from control.controllers.basic_controllers import Composite, Controller, Gravity
from control.estimators.forward import Forward
from control.simulation import Simulator
from control.environments.box import Box
from control.animation import Animator, Anim_2D
from control.noise import Gaussian_Noise, Noise
from control.estimators.base_estimator import Estimator

dt = 1./30 # 30 fps

################## set up objects ##########################

ball = Object_2D(M=1, dt=dt, init_state=[-0.5, 0.5, 0.03, 0.01],
                 delay=20)#, Q=numpy.eye(4)/40, R=numpy.eye(2)/40)

control = Composite([Controller(dim=2), Gravity(dim=2, gravity=-0.5)])
#estimator = Kalman_Filter(ball)
#estimator = Neural_KF()
estimator = Estimator(ball)
env = Box(ball)
forward = Forward(4, 2, ball.system_model, ball.meas_model, dt,
                  #noise=GaussianNoise(0.05), future_steps=10, c=10)
                  noise=Noise(), future_steps=10, c=10)

################# simulation functions #######################

def init_func(sim):
    sim.control = numpy.zeros(2)
    sim.ball_hist = []

def step_func(sim, i, time):
    sim.ball_hist.append(sim.plant.state)
    sim.control = sim.controller(sim.plant.state)
    sim.plant.step(sim.control, sim.dt)
    sim.env.step(sim.dt)

sim = Simulator({'plant':ball, 'controller':control, 'env':env},
                      dt=dt, init_func=init_func, step_func=step_func)

anim_objects = {'plant':Anim_2D()}
                #'forward':Forward_Object_2D_Animation,
                #'estimator':Estimator_Object_2D_Animation}

animator = Animator(title='Ball')
animator.axis.set_xlim(-1,1)
animator.axis.set_ylim(-1,1)
animator.display_text(['plant'], -1, 1)

steps = 1000
sim.run(1000)
animator.animate(data={'plant':sim.ball_hist}, anim_objects=anim_objects, frames=steps,
                 ms_per_frame=30)
