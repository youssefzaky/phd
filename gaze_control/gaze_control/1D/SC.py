import numpy as np
import nengo
from nengo.utils.function_space import Function, FunctionSpace

Bu, Bv, A = 1.4, 1.8, 3

def sacc_to_coll(theta, R):
    theta = np.deg2rad(theta)
    hor = Bu * np.log(np.sqrt(R**2 + A**2 + 2*A*R*np.cos(theta))) - Bu * np.log(A)
    ver = Bv * np.arctan2(R * np.sin(theta), R * np.cos(theta) + A)
    return np.array([hor, ver])

def coll_to_sacc(u, v):
    s_u, s_v = u / Bu, v / Bv
    R = A * np.sqrt(np.exp(2 * s_u) - 2 * np.exp(s_u) * np.cos(s_v) + 1)
    theta = np.rad2deg(np.arctan2(np.exp(s_u) * np.sin(s_v), (np.exp(s_u) * np.cos(s_v) - 1)))
    return np.array([theta, R])

def mesh_and_stack(*axes):
    grid = np.meshgrid(*axes)
    return np.vstack(map(np.ravel, grid)).T


# collicular map approximate limits: hor is 5mm, ver is -3mm to 3mm, derived from mapping above
hlimit, vlimit = 3.5, 1.5
Rmax = 50.
n_points_x, n_points_y = 100, 100
domain = mesh_and_stack(np.linspace(-1, hlimit + 1, n_points_x),
                        np.linspace(-vlimit - 1, vlimit + 1, n_points_y))
change_sigma = 10  #  movement fields larger than this value have a larger area

def Gaussian_2D(center_x, center_y):
    x, y = domain[:, 0], domain[:, 1]
    # map to saccade space, and set movement field size based on amplitude
    _, R = coll_to_sacc(center_x, center_y)
    # get a sigma between 0.2 and 0.8 based on the amplitude
    sigma = 0.2 * (1 - R / Rmax) + (R/ Rmax) * 0.8
    return np.exp(((x - center_x)**2 + (y - center_y)**2) / (-2 * sigma**2)) / (2 * np.pi * sigma **2)

dx = (hlimit / n_points_x)
dy = (2 * vlimit) / n_points_y
dA = dx * dy

# find center of the Gaussian bump
def center_mass(func):
    m = np.sum(func) * dA
    if m < 0.001: return 0, 0
    y = domain[:, 1]
    x = domain[:, 0]
    My = np.dot(func, x) * dA
    Mx = np.dot(func, y) * dA
    return My/m, Mx/m

def polar_to_cart(theta, R):
    theta = np.deg2rad(theta)
    return R * np.cos(theta), R * np.sin(theta)


def SC(n_neurons=1000, n_basis=10):
    gauss2D = Function(Gaussian_2D,
                       center_x=nengo.dists.Uniform(0, hlimit),
                       center_y=nengo.dists.Uniform(-vlimit, vlimit))

    fs = FunctionSpace(gauss2D, n_samples=n_neurons, n_basis=n_basis)

    def center_mass_vec(FS_vector):
        func = fs.reconstruct(FS_vector)
        return center_mass(func)

    def SC_output_fn(FS_vector):
        theta, R = coll_to_sacc(*center_mass_vec(FS_vector))
        return polar_to_cart(theta, R)

    net = nengo.Network(label="SC")

    with net:
        ens = nengo.Ensemble(n_neurons=n_neurons, dimensions=fs.n_basis)
        ens.encoders = fs.project(Function(Gaussian_2D,
                                           center_x=nengo.dists.Uniform(0, hlimit),
                                           center_y=nengo.dists.Uniform(-vlimit, vlimit)))

        eval_points = fs.project(Function(Gaussian_2D,
                                          center_x=nengo.dists.Uniform(0, hlimit),
                                          center_y=nengo.dists.Uniform(-vlimit, vlimit)))

        stimulus = fs.make_stimulus_node(Gaussian_2D, 2)
        nengo.Connection(stimulus, ens, synapse=None)
        net.stim_transform = nengo.Node(size_in=2, output=lambda t, R: sacc_to_coll(0, R[0] - R[1]))
        nengo.Connection(net.stim_transform, stimulus, synapse=None)

        net.output = nengo.Node(size_in=2, label='output')
        nengo.Connection(ens, net.output, function=SC_output_fn,
                         eval_points=eval_points)

    return net
