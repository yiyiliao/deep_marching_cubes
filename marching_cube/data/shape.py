# Generate a series of cubes 
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import uniform 

class Shape(object):
    """Sample points from toy primitives such as ellipsoid and cube"""
    def __init__(self, s=1, rx=0, ry=0, rz=0, x0=0, y0=0, z0=0, px=0, py=0, pz=0, shear=0):

        # scale
        self.s = s

        # rotation
        self.rx = rx 
        self.ry = ry 
        self.rz = rz 

        #translation
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        
        # projective
        self.px = px
        self.py = py
        self.pz = pz

        # shear
        Rshear = np.asarray([[1, uniform(0, shear), uniform(0, shear)],
                             [uniform(0, shear), 1, uniform(0, shear)], 
        	             [uniform(0, shear), uniform(0, shear), 1]]) 

        # homography
        self.R = np.eye(4)
        Rx = np.asarray([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)] ])
        Ry = np.asarray([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)] ])
        Rz = np.asarray([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        Rscale = np.eye(3)*s
        self.R[0:3, 0:3] = np.dot(np.dot(np.dot(np.dot(Rx, Ry), Rz), Rshear), Rscale)
        self.R[0:3, 3] = np.asarray([self.x0, self.y0, self.z0])
        self.R[3, 0:3] = np.asarray([self.px, self.py, self.pz])

    def unit_sample(self):
        """ randomly sample a point lying on a canonical shape,
        need to be overrided by subclass
        """
        p = np.randn(4)
        return p


    def random_sample(self, N, x_min=-1e+6, x_max=1e+6, y_min=-1e+6, y_max=1e+6, z_min=-1e+6, z_max=1e+6):
        """randomly sample N points lying on the faces of a randomly created primitive """
        pts = []

        phi_range = [np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)]
        theta_range = [np.random.uniform(0, np.pi), np.random.uniform(0, np.pi)]

        while len(pts) < N:
	    # firstly sample points from the canonical shape
            p_unit = self.unit_sample()
            if p_unit is None:
                continue
	    # then apply random transformation matrix
            p = np.transpose(np.dot(self.R, p_unit))
            p = p/p[3]
            if p[0]>x_min and p[0]<x_max and p[1]>y_min and p[1]<y_max and p[2]>z_min and p[2]<z_max:
                pts.append(p[0:3])

        pts = np.asarray(pts)

        return pts

