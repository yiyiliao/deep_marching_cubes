import numpy as np
from ellipsoid import Ellipsoid
from cube import Cube
import os
import h5py
import torch
from torch.autograd import Variable


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_batch(data, rnd, args):
    """Load a batch from the data blob"""
    # unpack the data blob
    pts, pts_gt, voxel = data

    pts_rnd = pts.index_select(0, rnd)
    pts_rnd_gt = pts_gt.index_select(0, rnd)
    if args.encoder_type == 'voxel':
        voxel_rnd = voxel.index_select(0, rnd)

    if args.encoder_type == 'point':
        net_input = pts_rnd
    elif args.encoder_type == 'voxel':
        net_input = voxel_rnd
    return net_input, pts_rnd, pts_rnd_gt


def load_data(args, dtype, phase='train'):
    """Load points and binary occupancy grid"""
    pts = load_pts(args, phase)
    pts, pts_gt, pts_removed = add_perturbation(pts, args)
    if args.noise_gt == 1:
        pts_gt = pts

    if args.encoder_type == 'voxel':
        tsdf = load_tsdf(args, phase)
        thres = 0.0
        voxel = (tsdf < thres).astype(np.float)
    else:
        # dummy voxel
        voxel = np.zeros(1)

    pts = Variable(torch.from_numpy(pts).type(dtype), requires_grad=False)
    pts_gt = Variable(torch.from_numpy(pts_gt).type(dtype), requires_grad=False)
    voxel = Variable(torch.from_numpy(voxel).type(dtype), requires_grad=False).unsqueeze(dim=1)

    # update args.num_train and args.num_val in case the data is loaded from cached files
    exec('args.num_%s = %d' % (phase, pts.shape[0]))

    return args, (pts, pts_gt, voxel)

def load_pts(args, phase):
    """Load only points """

    cached_file = args.cached_train if phase == 'train' else args.cached_val

    if os.path.isfile(cached_file):
        print "Loading cached data from %s..." % cached_file
        pts = np.load(cached_file)
    else:
        # the shapenet data should be downloaded in advance
        assert args.data_type != 'shapenet', \
           "No cached file found for shapenet! Please download the data at first."
        # the primitives can be created and saved before training
        pts = create_primitive_set(cached_file, phase, args)

    return pts

def load_tsdf(args, phase=''):
    """Load only tsdf, note that we only support shapenet data_type when
    considering standard 3D volumetric input at this moment
    """

    if phase == '':
        phase = args.phase

    cached_file = args.cached_train if phase == 'train' else args.cached_val
    cached_file = cached_file.replace('points_', 'voxel_')

    assert os.path.isfile(cached_file), \
       "No cached file found for shapenet! Please download the data at first."

    print "Loading cached data from %s..." % cached_file

    return np.load(cached_file)


def create_primitive_set(cached_file, phase, args):
    """Sample points from randomly generated primitives"""

    print "No cached data found, creating random primitives..."
    x_grids = np.arange(0, args.num_cells+1, args.len_cell)
    y_grids = np.arange(0, args.num_cells+1, args.len_cell)
    z_grids = np.arange(0, args.num_cells+1, args.len_cell)
    if phase == 'train':
        num_data = args.num_train
    else:
        num_data = args.num_val
    num_sample = args.num_sample
    dim = 3

    # randomly sampled points lying on the edge of the shape
    pts = np.zeros((num_data, num_sample, dim))
    
    i = 0
    while i < num_data:
        if args.verbose and np.mod(i, 10) == 0:
            print "creating sample: %d/%d" % (i, num_data)

        s = np.random.uniform(float(args.num_cells)*0.3, float(args.num_cells)*0.35)
        x0 = np.random.uniform(np.median(x_grids)-1, np.median(x_grids)+1)
        y0 = np.random.uniform(np.median(y_grids)-1, np.median(y_grids)+1)
        z0 = np.random.uniform(np.median(z_grids)-1, np.median(z_grids)+1)
        rx = np.random.uniform(0, np.pi*2)
        ry = np.random.uniform(0, np.pi*2)
        rz = np.random.uniform(0, np.pi*2)
        px = np.random.uniform(0, 0.1)
        py = np.random.uniform(0, 0.1)
        pz = np.random.uniform(0, 0.1)


        # For topology classification
        if args.data_type == 'ellipsoid' or (args.data_type == 'mix3d' and i <= num_data*1/2):
            shape = Ellipsoid(s=s, rx=rx, ry=ry, rz=rz, x0=x0, y0=y0, z0=z0, px=px, py=py, pz=pz)
        elif args.data_type == 'cube' or (args.data_type == 'mix3d' and i > num_data*1/2):
            shape = Cube(s=s, rx=rx, ry=ry, rz=rz, x0=x0, y0=y0, z0=z0, px=px, py=py, pz=pz)

        pts[i, :, :] = shape.random_sample(
                num_sample,
                x_grids.min(), x_grids.max(),
                y_grids.min(), y_grids.max(),
                z_grids.min(), z_grids.max())

        i += 1

    if not os.path.isdir(os.path.dirname(cached_file)):
        os.makedirs(os.path.dirname(cached_file))
    np.save(cached_file, pts)

    return pts

def add_perturbation(pts, args):
    """Add perturbation to the uniformly sampled data
    1. randomly remove points in a virtual cone if args.bias_level>0.0
    2. randomly sample points if args.num_sample is smaller than current
       number of points
    3. add noise is args.noise>0.0
    """

    pts_sampled = pts
    pts_removed = []
    max_removed_length = 0

    # incomplete observation, remove all points in a virtual cone
    if args.bias_level>0.0:
        print "generating incomplete observaton with level %f..." % (args.bias_level)

        h = args.num_cells
        r = args.num_cells*args.bias_level
        middle = np.ones((1, 3)) * args.num_cells/2

        pts_sampled = np.zeros((pts_sampled.shape[0], args.num_sample, pts_sampled.shape[2]))

        for i in range(pts.shape[0]):
            pts_i = pts[i, :, :] - middle

            mask = np.ones(pts_i.shape[0],)

            cone_dir = np.random.uniform(-1, 1, (3, 1))
            cone_dir = cone_dir/np.linalg.norm(cone_dir)

            cone_dist = np.dot(pts_i, cone_dir)
            mask[cone_dist.flatten() < 0] = 0

            cone_radius = (cone_dist/h) * r
            orth_dist = np.linalg.norm(pts_i - np.dot(cone_dist, cone_dir.transpose()), axis=1)
            mask[orth_dist > cone_radius.flatten()] = 0

            mask = (1-mask).astype(np.bool_)
            removed_i = pts_i[(1-mask).astype(np.bool_), :] + middle
            pts_removed.append(removed_i)
            if removed_i.shape[0] > max_removed_length:
                max_removed_length = removed_i.shape[0]

            pts_i = pts_i[mask, :]
            if args.num_sample <= pts_i.shape[0]:
                sample_ind = np.random.permutation(pts_i.shape[0])[0:args.num_sample]
            elif args.num_sample > pts_i.shape[0]:
                sample_ind = np.random.choice(pts_i.shape[0], args.num_sample)
            pts_sampled[i, :, :] = pts_i[sample_ind, :] + middle

        # expand pts_removed to the same dimension
        pts_removed_mtx = np.zeros((pts_sampled.shape[0],
                                    max_removed_length,
                                    pts_sampled.shape[2]))
        for i in range(pts_sampled.shape[0]):
            if not len(pts_removed[i]):
                pts_removed[i] = np.ones((1, 3))*-1
            pts_removed_mtx[i] = pts_removed[i][
                                    np.random.choice(pts_removed[i].shape[0],
                                    max_removed_length),
                                    :]
        pts_removed = pts_removed_mtx

    else:
        pts_removed = np.asarray([[]]*pts_sampled.shape[0])

    # sample points
    if args.num_sample<pts_sampled.shape[1]:
        if args.verbose:
            print "sampling %d/%d pts..." % (args.num_sample, pts.shape[1])
        pts_sampled = np.zeros((pts_sampled.shape[0],
                                args.num_sample,
                                pts_sampled.shape[2]))
        for i in range(pts.shape[0]):
            pts_i = pts[i, :, :]
            sample_ind = np.random.permutation(pts_i.shape[0])[0:args.num_sample]
            pts_sampled[i, :, :] = pts_i[sample_ind, :]

    # add noise
    if args.noise>0:
        if args.verbose:
            print "adding noise to the data, noise level %.02f" % args.noise
        pts_sampled = pts_sampled + np.random.normal(0, args.noise*args.num_cells/2, pts_sampled.shape)

    return pts_sampled, pts, pts_removed
