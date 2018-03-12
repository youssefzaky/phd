import numpy
from matplotlib import pyplot as plt
from matplotlib import animation

################# Aniimator class ###################

class Animator(object):
    """
    Given data and objects that specify how to animate that data,
    this class is responsible for running the animation.

    Parameters:
    title: the title of the animation
    """

    def __init__(self, title, xlim=1, ylim=1, zlim=1, three_d=False):
        self.title = title
        self.display = []
        self.fig = plt.figure()
        self.fig.suptitle(self.title)
        if three_d:
            from mpl_toolkits.mplot3d import Axes3D
            self.axis = self.fig.add_subplot(111, projection='3d')
            self.axis.set_zlim(-zlim, zlim)
        else:
            self.axis = self.fig.add_subplot(111)

        self.axis.set_xlim(-xlim, xlim)
        self.axis.set_ylim(-ylim, ylim)
        self.text = ''
        self.anim_objects = {}

    def display_text(self, list, x=-1, y=1):
        """A list containting some the data keys that should be displayed as text."""
        self.text_line = self.axis.text(x, y, '', va='top')
        self.display = list

    def animate(self, data, anim_objects, frames, repeat_delay=20,
                ms_per_frame=10, save=False):
        """Parameters:

        data: a dictionary consisting of (name, data) pairs
        anim_classes: a dictionary consisting of (name, class) pairs. The names must be
        a subset of the names in data, and the class specifies how the lines are drawn
        frames: the number of frames in the animation
        repeat_delay: how long to wait before repeating the animation
        ms_per_frame: how many ms to spend per frame
        """

        self.anim_objects = anim_objects

        # call make_lines of all animation objects
        for anim_object in anim_objects.itervalues():
            anim_object.make_lines(self.axis)

        # make the animation init function
        def anim_init():

            lines = []
            for key in anim_objects.keys():
                lines.extend(self.anim_objects[key].anim_reset())

            if hasattr(self, 'text_line'):
                self.text_line.set_text('')
                lines.extend([self.text_line])
            return lines

        # make the animation animate function
        def anim_animate(i):

            self.text = ''
            lines = []
            for key in anim_objects.keys():
                index = min(i, len(data[key]) - 1)
                try:
                    lines.extend(self.anim_objects[key].anim_update(data[key][index]))
                except IndexError:
                    print("Index %s is out of range "
                           "for data of shape %s belonging to key '%s'"
                          "" % (index, data[key].shape, key))

            if hasattr(self, 'text_line'):
                #update text every 10 steps, otherwise numbers go by too fast
                if i % 10 == 0:
                    self.text_line.set_text(self.make_info_text(data, i))
                lines.extend([self.text_line])
            return lines

        #need to keep a handle to this
        anim = animation.FuncAnimation(self.fig, anim_animate,
                                       repeat_delay=repeat_delay,
                                       init_func=anim_init, frames=frames,
                                       interval=ms_per_frame, blit=True)

        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
        if save:
            anim.save('name.mp4', extra_args=['libx264', '-vcodec'] )
        else:
            plt.show()

    def make_info_text(self, data, i):
        """Make text info from the data that is being animated."""

        for key in data.keys():
            if key in self.display:
                self.text += '\n' + key + ' ' +  numpy.array_str(data[key][i])
        return self.text


############### Anim_2D class #######################

class Anim_2D(object):
    """Defualt animation interface for 2D lines"""

    def __init__(self, **line_kwargs):
        if line_kwargs != {}:
            self.line_kwargs = line_kwargs
        else:
            self.line_kwargs = {'color':'#888888',
                                'marker':'o',
                                'markersize':6}

    def make_lines(self, ax):
        self.line = ax.plot([], [], **self.line_kwargs)[0]

    def anim_reset(self):
        self.line.set_data([], [])
        return [self.line]

    def anim_update(self, data):
        self.line.set_data(data[0], data[1])
        return [self.line]

############# Anim_Hist class #####################

class Anim_Hist(Anim_2D):
    """Interface for animating a trail of 2D line"""

    def anim_update(self, data):
        self.line.set_data(data[0, :], data[1, :])
        return [self.line]

##################################################
