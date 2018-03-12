import numpy


class Simulator(object):
    """Simulates the objects that are passed to it.

    Parameters
    ----------
    objects : dict
          Keys are names of the objects, values are the objects. The
          objects can then be accessed by their names

    dt      : float, optional
          Simulator timestep

    init_func : callable
         Function to be executed prior to simulation, receives the
         simulator as an argument

    step_func: callable
         Function to be executed every step of the simulation,
         receives the simulator as an argument
    """

    def __init__(self, objects, init_func, step_func, dt=0.001):
        self.objects = objects
        for name, instance in objects.items():
            setattr(self, name, instance)

        self.dt = dt

        self.elapsed_time = 0

        self.step_func = step_func
        self.init_func = init_func
        self.init = False
        self.exp_list = []
        self._data = {}

    def run(self, steps=1, time=None):
        """Note: if specified, time overrides steps"""

        # call init only one time
        if self.init is False:
            self.init_func(self)
            self.init = True

        if time is None:
            self.steps = steps
        else:
            self.steps = numpy.around(time / self.dt).astype(int)

        for i in range(self.steps):

            for exp in self.exp_list:
                self._data[exp].append(eval('self.' + exp))

            self.step_func(self, i, self.elapsed_time)

            self.elapsed_time += self.dt

    @property
    def trange(self):
        return numpy.arange(self.dt, self.elapsed_time, self.dt)

    def data(self, exp, to_array=True):
        if to_array:
            return numpy.array(self._data[exp])
        else:
            return self._data[exp]

    def record(self, *exp_list):
        self.exp_list = exp_list
        for exp in exp_list:
            self._data[exp] = []


class Experiment(object):
    """Runs experiments using the given simulator.

    Parameters
    ----------

    sim: Simulator object
        Used for running the experiments
    duration: float
        The length of each trial in seconds
    n_trials: int
        Number of trials
    """

    def __init__(self, sim, duration, n_trials, callback=None):
        self.sim = sim
        self.duration = duration
        self.n_trials = n_trials

        self.sim.init = True # override init call in sim.run above
        self.interval = duration / sim.dt
        self.trange = numpy.arange(0, duration, sim.dt)
        self.total_time = duration * n_trials
        self.total_steps = numpy.around(self.total_time / sim.dt).astype(int)
        self.callback = callback

    def run(self):
        for _ in range(self.n_trials):
            self.sim.init_func(self.sim)
            self.sim.run(time=self.duration)
            if self.callback is not None: self.callback(self.sim)

    def data(self, exp, trial=None, to_array=True):
        i = trial
        interval = self.interval

        if trial is None:
            return self.sim.data(exp, to_array=to_array)
        else:
            return self.sim.data(exp, to_array=to_array)[i*interval:(i+1) * interval]
