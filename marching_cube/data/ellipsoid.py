# Generate a series of ellipoids 
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import uniform 
from shape import Shape

class Ellipsoid(Shape):

    def unit_sample(self):
        """ randomly sample a point lying on a unit ellipsoid """ 
        p = np.zeros(4)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        p[0] = np.sin(theta) * np.cos(phi) 
        p[1] = np.sin(theta) * np.sin(phi) 
        p[2] = np.cos(theta) 
        p[3] = 1
        return p


if __name__=="__main__":
    ellipsoid = Ellipsoid(s = np.random.uniform(0.95,1.05),
		x0 = np.random.uniform(2,4),
		y0 = np.random.uniform(2,4),
		z0 = np.random.uniform(2,4),
		rx = np.random.uniform(0, np.pi*2),
		ry = np.random.uniform(0, np.pi*2),
		rz = np.random.uniform(0, np.pi*2),
		px = np.random.uniform(0, 0.1),
		py = np.random.uniform(0, 0.1),
		pz = np.random.uniform(0, 0.1),
		shear = 0.0)
    pts = ellipsoid.random_sample(3000)
    print pts
    print pts.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_aspect('equal') 
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    plt.show()
