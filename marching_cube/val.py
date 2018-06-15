import numpy as np
import torch
from torch.autograd import Variable
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'model/cffi'))
import matplotlib as mpl
mpl.use('Agg')
from utils.visualize import save_occupancy_fig, save_mesh_fig
from utils.config import parse_args
from model.table import get_accept_topology
from model.loss import Loss
from data.data_loader import load_data, get_batch 

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor


def run_val(model, loss_obj, data_val, args, phase='train'):
    """ Test with the trained model

    Input:
        model: the Deep Marching Cubes model
        loss_obj: the loss instance
        data_val: validation data blob, including points and voxel grid
        args: configuration arguments
        phase: 'train' or 'val'
    """


    max_prob = []
    # evaluation
    loss_eval = 0
    for itest in range(args.num_val):
        sys.stdout.write('.')
        sys.stdout.flush()

        itest_ = Variable(dtype_long([itest]))
        net_input, pts_rnd, _ = get_batch(data_val, itest_, args)

        offset, topology, occupancy = model(net_input)

        loss = loss_obj.loss_eval(offset[0], topology[0], pts_rnd[0])

        loss_eval += loss.data.cpu()[0]


        interval = args.verbose_interval 
        if phase == 'val':
            interval = 1
        if np.mod(itest, interval) == 0:
            topology_fused = topology[-1].data.cpu().numpy()
            topology_fused = np.maximum(topology_fused[:, 0:128],
                                        topology_fused[:, 256:127:-1])
            topology_fused = topology_fused[:, get_accept_topology()]
            save_occupancy_fig(
                    pts_rnd[-1].data.cpu().numpy(),
                    occupancy[-1].data.cpu().numpy(),
                    loss_obj.x_grids,
                    loss_obj.y_grids,
                    loss_obj.z_grids,
                    itest, args, 'val')

            topology_vis = topology[:, :, loss_obj.visTopology]

            save_mesh_fig(
                    pts_rnd[-1].data.cpu().numpy(),
                    offset[-1],
                    topology_vis[-1],
                    loss_obj.x_grids,
                    loss_obj.y_grids,
                    loss_obj.z_grids,
                    itest, args, 'val')


    print('')
    return loss_eval

if __name__ == '__main__':

    # parse args
    args = parse_args()

    # load data
    args, data_val = load_data(args, dtype, 'val')

    # setup loss object
    loss_obj = Loss(args)

    # initialize the model
    assert(os.path.isfile(args.model))
    print "Validating with snapshotted model %s ..." % args.model
    deep_marching_cubes = torch.load(args.model)
    if torch.cuda.is_available():
        deep_marching_cubes.cuda()

    # validation
    loss = run_val(deep_marching_cubes, loss_obj, data_val, args, 'val')
    print('============== average loss:%f' % (loss/args.num_val))


    print 'Done!' 
