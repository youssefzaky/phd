from box import Box
import collections
import numpy

class Target(Box):
    """
    An environment for target tasks. Defines target selection,
    error computations.

    Parameters:
    (Optional) targets: an iterable of targets or an integer. If an integer,
    then it's the number of targets that will be simulated. If it's an iterable
    then it's the list of targets. The latter, in combination with steps_per_target
    or time_per_target, allows the specification of trajectories.
    (Optional) steps_per_target: number of time steps the plant can spend reaching
    for the current target
    (Optional) time_per_target: same as above, but specified in time
    """

    def __init__(self, targets=10, steps_per_target=20, dt=0.001,
                 time_per_target=None, **kwargs):
        if time_per_target is None:
            self.steps_per_target = steps_per_target
        else:
            self.steps_per_target = numpy.round(time_per_target / dt)

        self.targets = numpy.matrix(targets)
        self.total_steps = 0
        self.target_count = 0
        super(Target, self).__init__(**kwargs)
        self.current_target = self.get_next_target()

    def get_next_target(self):
        # keep returning the last target if more are
        # required than given
        index = min(self.target_count, len(self.targets) - 1)
        target = self.targets[index]
        self.target_count += 1
        return target.T

    def step(self, dt):
        self.total_steps += 1
        if self.total_steps % self.steps_per_target == 0:
            self.current_target = self.get_next_target()
