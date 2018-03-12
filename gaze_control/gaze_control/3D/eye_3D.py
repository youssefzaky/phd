import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from Quaternion import Quat

"""TODO:

- add animation
- figure out units for coordinate system, forces, mass etc
- desired velocity to desired joint torque (involves CB?)
- desired joint torque to desired muscle torque
- muscle pulling directions and muscle to joint torque
- foveate using quaternions
- neural circuit for each muscle pair
- neural circuit for VOR
- how to convert retinal error to desired quaternion?
- sensory axes to non-orthogonal motor axes
- what kind of adaptations are there? saccadic adaptation, VOR gain
- control: cross-product, quaternion (Tweed), kinematic goals (Dehmi)
- head muscles
"""

dt = 0.001
q_id = Quat([0, 0, 0, 1]) # identity quaternion
t_id = np.zeros(3) # identity translation

def solve_matrix(A, B):
    """Obtain the matrix that rotates the 3D vector A to 3D vector B
    through the plane passing through both.

    Reference:
    http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/180436#180436
    """

    A /= np.linalg.norm(A)
    B /= np.linalg.norm(B)
    c = - np.cross(A, B)
    d = np.dot(A, B)

    G = np.array([[d, - np.linalg.norm(c), 0], [np.linalg.norm(c), d, 0], [0, 0, 1]])
    v = B - d * A
    v /= np.linalg.norm(v)
    F = np.array([A, v, c])
    R = np.dot(np.dot(np.linalg.inv(F), G), F)
    return R

def derivative(x, dt):
    return (x[1:] - x[:-1]) / dt

def rotate_vec(v, w):
    """Vector integration from angular velocity"""
    vel = np.cross(w, v)
    return v + dt * vel

def plot_vector(ax, end, direction, **kwargs):
    """Helper function for plotting vectors"""
    ax.quiver([end[0]], [end[1]], [end[2]], [direction[0]],
              [direction[1]], [direction[2]], length=np.linalg.norm(direction),
              **kwargs)

def quat_rotate(q, x):
    """Quaternion by 3D vector multiplication"""
    X = Quat([x[0], x[1], x[2], 0])
    return (q * X * q.inv()).q[:3]

def transform(x, translation=t_id, rotation=q_id):
    """Translation (3D vector) and Rotation (Quaternion) of a vector"""
    return translation + quat_rotate(rotation, x)

def integ_quat(q, w):
    """Quaternion integration from angular velocity"""
    w = np.array(w) / 2.0
    q_dot = Quat([w[0], w[1], w[2], 0]) * q
    temp = q.q + q_dot.q * dt
    return Quat(temp / np.linalg.norm(temp))


class Sphere(object):
    """This is a static object, mainly for visualization purposes"""

    def __init__(self, radius=1, trans=t_id, rotation=q_id, trans_cs=t_id,
                 rotation_cs=q_id, objs=[], m=1):
        self.r = radius
        self.trans = np.array(trans) # translation relative to containing frame
        self.quat = rotation # orientation relative to containing frame
        self.update_space(trans_cs, rotation_cs)
        self.objs = objs
        for obj in self.objs:
            obj.update_space(self.trans_s, self.quat_s)

        self.inertia = np.eye(3) * (2.0/5) * m * self.r**2
        self.mom = np.zeros(3) # angular momentum
        self.vel = np.array([0, 0, 0]) # angular velocity

        n = 0.001 # viscosity of surrounding tissue
        self.k = 6 * np.pi * n * self.r # viscosity constant for a sphere
        self.s = 1 # spring constant

        # parametrization of a sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        # spherical surface
        self.x = self.r * np.outer(np.cos(u), np.sin(v))
        self.y = self.r * np.outer(np.sin(u), np.sin(v))
        self.z = self.r * np.outer(np.ones(np.size(u)), np.cos(v))

        # initial body fixed axis, defined relative to the containing object
        self.bfx = np.array([1, 0, 0])
        self.bfy = np.array([0, 1, 0])
        self.bfz = np.array([0, 0, 1])

        # initial body fixed points, defined relative to the contatining object
        self.back = self.bfx * -self.r
        self.front = self.bfx * self.r
        self.side = self.bfy * self.r
        self.up = self.bfz * self.r

        self.back_traj = [transform(self.back, self.trans_s, self.quat_s)]

    def update_space(self, trans_cs, rotation_cs):
        # translation relative to space
        self.trans_s = np.array(trans_cs) + transform(self.trans, rotation=rotation_cs)
        self.quat_s = rotation_cs * self.quat # orientation relative to space

    def apply_vel(self, ang_vel, trans_cs, rotation_cs, timesteps=1):
        """Rotate sphere with the given velocity for number of timesteps.
        The rotation happens relative to a coordinate system fixed to the
        containing frame and centered on the sphere"""
        ang_vel = np.array(ang_vel)
        for _ in range(timesteps):
            self.back_traj.append(transform(self.back, self.trans_s, self.quat_s))
            self.quat = integ_quat(self.quat, ang_vel)
            self.update_space(trans_cs, rotation_cs)
            for obj in self.objs:
                obj.update_space(self.trans_s, self.quat_s)
        # can also compute orientation changes wrt space after the loop

    def apply_torque(self, torque, trans_cs, rotation_cs, timesteps=1):
        for _ in range(timesteps):
            torque += self.internal_torque()
            self.mom += dt * np.array(torque) # integrate torque to get momentum
            O = self.quat.transform # orientation matrix
            # inverse of inertia tensor in container fixed coordinates
            J_inv_h = np.dot(O, np.dot(np.linalg.inv(self.inertia), O.T))
            self.vel = np.dot(J_inv_h, self.mom) # angular velocity
            self.apply_vel(self.vel, trans_cs, rotation_cs)

        self.mom = 0

    def internal_torque(self):
        t_visc = -self.k * self.vel
        t_spring = self.s * (q_id * self.quat.inv()).q[0:3]
        torque = t_spring + t_visc
        print 't_spring:', t_spring / np.linalg.norm(t_spring)
        print 't_visc:', t_visc / np.linalg.norm(t_visc)
        return torque

    def control(self, desired):
        error = desired * self.quat.inv()
        vel = self.gain * error[:3]
        # counteracts spring torque
        spring_e = -self.s * (q_id * self.quat.inv()).q[0:3]
        vel = vel + spring_e
        mom = np.dot(self.inertia, vel)
        torque = (self.mom - mom) / dt
        return torque

    def plot(self, ax, target=None, color='y'):
        """Plot in local coordinates in space coordinates"""
        translation = self.trans_s
        rotation = self.quat_s

        # translate to correct spatial position relative to container
        x = self.x + translation[0]
        y = self.y + translation[1]
        z = self.z + translation[2]
        ax.plot_surface(x, y, z, color=color, linewidth=0,
                        antialiased=False, alpha=0.2)

        # transform to space coordinate system
        front = transform(self.front, translation, rotation)
        side = transform(self.side, translation, rotation)
        up = transform(self.up, translation, rotation)

        # rotate body fixed axes
        bfx = transform(self.bfx, rotation=rotation)
        bfy = transform(self.bfy, rotation=rotation)
        bfz = transform(self.bfz, rotation=rotation)

        plot_vector(ax, front, self.r * bfx, color='k')
        plot_vector(ax, side, self.r * bfy, color='k')
        plot_vector(ax, up, self.r * bfz, color='k')

        back_traj = np.array(self.back_traj)
        self.back_traj = [self.back_traj[-1]]
        ax.plot(back_traj[:, 0], back_traj[:, 1],
                back_traj[:, 2], color='r')

    def error_2D(self, target):
        """Project onto the eye-fixed axes to obtain retinal error relative
        eye"""
        translation = self.trans_s
        rotation = self.quat_s
        back = transform(self.back, translation, rotation)
        bfy = transform(self.bfy, rotation=rotation)
        bfz = transform(self.bfz, rotation=rotation)
        inter = self.intersection(target)
        error_3D = inter - back
        proj_y = np.dot(error_3D, bfy)
        proj_z = np.dot(error_3D, bfz)
        error_2D = np.array([proj_y, proj_z])
        return error_2D

    def intersection(self, target):
        """Calculates the point of intersection of the straight line from the
        target to the entry point with the back of the sphere, i.e, the retina.
        This is calculated with respect to the space coordinate frame"""
        t = target
        l = self.trans_s
        front = transform(self.front, self.trans_s, self.quat_s)
        d = front - target
        a = d[0]**2 + d[1]**2 + d[2]**2
        b = (2*t[0]*d[0] - 2*d[0]*l[0]) + (2*t[1]*d[1] - 2*d[1]*l[1])  \
        + (2 *t[2]*d[2] - 2*d[2]*l[2])
        c = target[0]**2 + target[1]**2 + target[2]**2 + l[0]**2 + l[1]**2 + l[2]**2 \
        - 2*t[0]*l[0] - 2*t[1]*l[1] - 2*t[2]*l[2] - self.r**2
        return target + np.amax(np.roots([a, b, c])) * d


class Head(Sphere):

    def __init__(self, **kwargs):
        trans_s = kwargs.get('trans')
        rotation_s = kwargs.get('rotation')
        eye_l = Eye('l', trans=[1.7, 1, 1.5],
                    trans_cs=trans_s,
                    rotation_cs=rotation_s, m=0.1)
        eye_r = Eye('r', trans=[1.7, -1, 1.5],
                    trans_cs=trans_s,
                    rotation_cs=rotation_s, m=0.1)
        super(Head, self).__init__(objs=[eye_l, eye_r], **kwargs)
        self.gain = 5


class Eye(Sphere):

    def __init__(self, side, **kwargs):
        super(Eye, self).__init__(**kwargs)
        self.gain = 2

        # self.mr = Muscle([0, -1, 0], [-1, 0, 0], f=2)
        # self.lr = Muscle([0, 1, 0], [-1, 0, 0,], f=2)
        # self.sr = Muscle([0, 1, 0], [-1, 0, 0,], f=2)
        # self.ir = Muscle([0, 1, 0], [-1, 0, 0,], f=2)
        # self.so = Muscle([0, 1, 0], [-1, 0, 0,], f=2)
        # self.io = Muscle([0, 1, 0], [-1, 0, 0,], f=2)

    def muscle_to_joint_torque(self):
        """for simulation. see shadmehr book"""


    def joint_to_muscle_torque(self, joint_torque):
        """used for control"""


class Eye_Head(object):

    def __init__(self):

        self.head = Head(radius=3, trans=np.array([1, 1, 1]), rotation=q_id)

    def plot(self, target=None, vectors=None):

        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection='3d')

        plot_vector(self.ax, [1, 0, 0], [1, 0, 0])
        plot_vector(self.ax, [0, 1, 0], [0, 1, 0])
        plot_vector(self.ax, [0, 0, 1], [0, 0, 1])

        if vectors is not None:
            for v in vectors:
                plot_vector(self.ax, v, v, color='r')

        self.head.plot(self.ax)
        self.head.objs[0].plot(self.ax, color='g')
        self.head.objs[1].plot(self.ax, color='g')

        if target is not None:
            self.ax.plot([target[0]], [target[1]], [target[2]], marker='o',
                         markersize=10, color='g')
            ret_l = self.head.objs[0].intersection(target)
            ret_r = self.head.objs[1].intersection(target)
            self.ax.plot([ret_l[0]], [ret_l[1]], [ret_l[2]], marker='o',
                         markersize=7)
            self.ax.plot([ret_r[0]], [ret_r[1]], [ret_r[2]], marker='o',
                         markersize=7)

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        plt.show()

    def foveate_cross(self, target, plot=True):
        # eye curvature
        # x = self.r - np.sqrt(np.maximum((self.r ** 2) - (error_2D[0] ** 2) - (error_2D[1] ** 2), 0))
        # ret = [x, error_2D[0], error_2D[1]]
        eye_l = self.head.objs[0]
        eye_r = self.head.objs[1]
        error_2D_l = eye_l.error_2D(target)
        error_2D_r = eye_r.error_2D(target)
        error_2D_h = self.head.error_2D(target)
        error_3D_l = np.array([0, error_2D_l[0], error_2D_l[1]])
        error_3D_r = np.array([0, error_2D_r[0], error_2D_r[1]])
        error_3D_h = np.array([0, error_2D_h[0], error_2D_h[1]])
        w_ee_l = np.cross(error_3D_l, eye_l.bfx)
        w_ee_r = np.cross(error_3D_r, eye_r.bfx)
        w_h = -np.cross(self.head.bfx, error_3D_h)

        w_hs = quat_rotate(self.head.quat, w_h)
        w_eh_l = quat_rotate(eye_l.quat, w_ee_l)
        w_eh_r = quat_rotate(eye_r.quat, w_ee_r)

        while np.linalg.norm(eye_l.error_2D(target)) > 0.01:
            eye_l.apply_vel(w_eh_l, self.head.trans_s, self.head.quat_s)

        while np.linalg.norm(eye_r.error_2D(target)) > 0.01:
            eye_r.apply_vel(w_eh_r, self.head.trans_s, self.head.quat_s)

        self.plot(target)
        plt.show()

        while np.linalg.norm(self.head.error_2D(target)) > 0.01:
            self.head.apply_vel(w_hs, t_id, q_id)
            vel_el = quat_rotate(eye_l.quat.inv() * self.head.quat.inv(), -w_h)
            vel_er = quat_rotate(eye_r.quat.inv() * self.head.quat.inv(), -w_h)
            eye_l.apply_vel(vel_el, self.head.trans_s, self.head.quat_s)
            eye_r.apply_vel(vel_er, self.head.trans_s, self.head.quat_s)

        self.plot(target)
        plt.show()

    def foveate_quat(self, target):
        eye_r = self.head.objs[1]
        ret = eye_r.intersection(target)
        d = eye_r.front - ret
        q_d = Quat(solve_matrix(eye.bfx, d))
        q_d = Quat(q_d)
        q = eye_r.quat
        E = q_d * q.inv()

        fovea_traj = []

        while np.linalg.norm(E.q - [0, 0, 0, 1]) > 0.01:
            w = E.q[:3]
            eye_r.apply_vel(w/2, self.head.trans_s, self.head.quat_s)
            q = integ_quat(q, w)
            E = q_d * q.inv()
            fovea_traj.append(eye_r.back)
        traj = np.array(fovea_traj)
        eye_r.quat = q


class Muscle(object):
    """Parameters:
    point: initial insertion point (3D)
    direc: initial pulling direction (3D)
    f: initial force (scalar)

    Vectors are in a container fixed frame centered on the object.
    """

    def __init_(self, point, direc, f):
        self.point = point
        self.direc = direc
        self.f = f
        self.alpha_1 = 0.004
        self.alpha_2 = 1

    def torque(self):
        return np.cross(self.point, self.torque)

    def update_f(self, act):
        self.f += dt * (act - self.alpha_2 * self.f) / self.alpha_1

    def update_point(self, quat):
        self.point = quat_rotate(quat, self.point)
