import numpy as np
import matplotlib.pyplot as plt


D = 28

def plot_image(image, dim=D):
    plt.figure()
    plt.imshow(image.reshape((dim, dim)), vmin=-1, vmax=1, cmap='gray')
    plt.xticks([])
    plt.yticks([])


def plot_images(images, dim=D):
    """Images should be of shape (n_images, dim * dim)"""
    N = images.shape[0]
    plt.figure(figsize=(10,8))
    s = 20
    for i in range(N):
        w = i%s
        h = i/s
        plt.imshow(images[i].reshape((dim, dim)), extent=(w, w+0.95, h, h+0.95),
                   interpolation='none', cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlim((0, s))
    plt.ylim((0, N/s))


# the real part of a gabor filter defined on a local part of an image
def real_gabor(points, psi, gamma, sigma, lambd, theta, x_offset, y_offset):
    x = points[:, 0]
    y = points[:, 1]
    c_x = x - x_offset
    c_y = y - y_offset

    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    xTheta = c_x * cosTheta  + c_y * sinTheta
    yTheta = -c_x * sinTheta + c_y * cosTheta
    e = np.exp( -(xTheta**2 + yTheta**2 * gamma**2) / (2 * sigma**2) )
    cos = np.cos(2 * np.pi * xTheta / lambd + psi)
    return e * cos


# the imaginary part
def imag_gabor(points, psi, gamma, sigma, lambd, theta, x_offset, y_offset):
    x = points[:, 0]
    y = points[:, 1]
    c_x = x - x_offset
    c_y = y - y_offset

    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    xTheta = c_x * cosTheta  + c_y * sinTheta
    yTheta = -c_x * sinTheta + c_y * cosTheta
    e = np.exp( -(xTheta**2 + yTheta**2 * gamma**2) / (2 * sigma**2) )
    sin = np.sin(2 * np.pi * xTheta / lambd + psi)
    return e * sin

sigma1 = 0.1 # bigger
sigma2 = 0.07 # smaller

def center_on(points, center_x, center_y):
    x = points[:, 0]
    y = points[:, 1]
    c_x = x - center_x
    c_y = y - center_y
    e1 = np.exp( -(c_x**2 + c_y**2) / (2 * sigma1**2) )
    e2 = 2 * np.exp( -(c_x**2 + c_y**2) / (2 * sigma2**2) )
    return e2 - e1

def center_off(points, center_x, center_y):
    x = points[:, 0]
    y = points[:, 1]
    c_x = x - center_x
    c_y = y - center_y
    e1 = np.exp( -(c_x**2 + c_y**2) / (2 * sigma1**2) )
    e2 = 2 * np.exp( -(c_x**2 + c_y**2) / (2 * sigma2**2) )
    return e1 - e2
