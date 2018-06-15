import mcubes
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

input_file_low = "/is/ps2/sdonne/Desktop/TSDF_32.npy"
output_file_mesh_low = "/is/ps2/sdonne/Desktop/TSDF_32.off"

input_file = "/is/ps2/sdonne/Desktop/TSDF_256.npy"
output_file_mesh = "/is/ps2/sdonne/Desktop/TSDF_256.off"
output_file_pts  = "/is/ps2/sdonne/Desktop/TSDF_256_points.off"


def write_to_off(vertices, faces, filename):
    f = open(filename, 'w')
    f.write('OFF\n')
    # f.write('# Author: Yiyi\n')
    # f.write('# \n')

    n_vertice = vertices.shape[0]
    n_face = faces.shape[0]
    f.write('%d %d 0\n' % (n_vertice, n_face))
    for nv in range(n_vertice):
        ## !!! exchange the coordinates to match the orignal simplified mesh !!!
        ## !!! need to check where the coordinate is exchanged !!!
        f.write('%f %f %f\n' % (vertices[nv, 1], vertices[nv, 0], vertices[nv, 2]))
    for nf in range(n_face):
        f.write('3 %d %d %d\n' % (faces[nf, 0], faces[nf, 1], faces[nf, 2]))

def sample_mesh(vertices, faces, npoints):
    # note: we do assume the mesh is built from triangles, but both marching cubes implementations fulfill this
    n_faces = faces.shape[0]
    # calculate all face areas, for total sum and cumsum
    # sample uniformly into the fractional size array, and then sample uniformly in that triangle
    v0s = vertices[faces[:,0],:]
    v1s = vertices[faces[:,1],:]-v0s
    v2s = vertices[faces[:,2],:]-v0s
    
    areas = np.power(np.sum(np.power(np.cross(v2s,v1s),2),1), 0.5)
    frac_idces = np.cumsum(areas / np.sum(areas))

    triangle_fracs = np.random.rand(npoints)
    triangle_idces = np.searchsorted(frac_idces,triangle_fracs)

    v1_fracs = np.random.rand(npoints,1)
    v2_fracs = np.random.rand(npoints,1)
    frac_out = (v1_fracs + v2_fracs > 1)
    v1_fracs[frac_out] = 1-v1_fracs[frac_out]
    v2_fracs[frac_out] = 1-v2_fracs[frac_out]

    points = v0s[triangle_idces,:] + v1_fracs * v1s[triangle_idces,:] + v2_fracs * v2s[triangle_idces,:]

    return points

if __name__=="__main__":

    sdf = np.load(input_file)
    sdf_low = np.load(input_file_low)

    try:
        import mcubes
        marching_cubes_method = "mcubes"
    except ImportError as ex:
        print("Could not import mcubes, switching to scikit measure")
        try:
            from skimage import measure
            marching_cubes_method = "scikit_measure"
        except ImportError as ex:
            print("Could not load scikit measure either. Please install one of both options.")
            exit(1)
    
    if marching_cubes_method == "mcubes":
        vertices, faces = mcubes.marching_cubes(sdf, 0)
        vert_low, f_low = mcubes.marching_cubes(sdf_low, 0)
    elif marching_cubes_method == "scikit_measure":
        vertices, faces, _, _ = measure.marching_cubes_lewiner(sdf, 0)
        vert_low, f_low, _, _ = measure.marching_cubes_lewiner(sdf_low, 0)
    
    # rescale to the correct area in space, so that they overlap
    vertices = vertices / 256 + 0.5/256 - 0.5
    vert_low = vert_low / 32 + 0.5/32 - 0.5
    
    write_to_off(vertices,faces,output_file_mesh)
    write_to_off(vert_low,f_low,output_file_mesh_low)
    write_to_off(sample_mesh(vertices,faces,3000),np.zeros([0,3]),output_file_pts)
