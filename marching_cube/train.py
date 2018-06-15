# Train and validation of the deep marching cubes

import torch
from torch.autograd import Variable
import numpy as np
import os
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.getcwd(), 'model/cffi'))
from utils.visualize import save_occupancy_fig, save_mesh_fig
from utils.config import parse_args
from model.dmc import DeepMarchingCube
from model.loss import Loss
from model.table import get_accept_topology
from data.data_loader import load_data, get_batch
from val import run_val

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor


def run_train_val(model, optimizer, loss_obj, data_train, data_val, args):
    """ Train the model

    Input:
        model: the Deep Marching Cubes model
        optimizer: the optimizer for parameters of model
        loss_obj: the loss instance
        data_train: training data blob, including points and voxel grid
        data_val: validation data blob, including points and voxel grid
        args: configuration arguments
    """

    loss_epochs = []
    loss_evals = []

    for iepoch in range(curr_epoch+1, args.epoch):

        rnd_list = np.random.permutation(args.num_train)

        num_batch = np.ceil(len(rnd_list)/float(args.batchsize)).astype(np.int)

        loss_epoch = 0
        for ibatch in range(num_batch):

            rnd = rnd_list[ibatch*args.batchsize : np.min((len(rnd_list),
                           (ibatch+1)*args.batchsize))]
            rnd = Variable(dtype_long(rnd))

            # get the batch of the data
            net_input, pts_rnd, pts_rnd_gt = get_batch(data_train, rnd, args)

            # forward pass
            offset, topology, occupancy = model(net_input)

            # get loss
            loss, loss_stages = loss_obj.loss_train(offset, topology,
                                pts_rnd_gt, occupancy)
            loss_epoch += loss.data[0]*args.batchsize

    	    # update weights
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if args.verbose and np.mod(ibatch, args.verbose_interval) == 0:
                loss_stages_str = ['loss%d:%f' % (i, l) for i, l in enumerate(loss_stages)]
                print('id:%d, loss:%f, %s ' % (ibatch, loss.data[0], ', '.join(loss_stages_str))) 

        print('====== epoch:%d, average loss:%f ======' % (iepoch, loss_epoch/args.num_train))
        loss_epochs.append(loss_epoch/args.num_train)

        # save model
        fname = '%s_noise%.02f_lr%f' % (args.data_type, args.noise, args.learning_rate)
        if args.save_model and np.mod(iepoch, args.snapshot) == 0:
            torch.save(model, os.path.join(args.output_dir, 'snapshot_%s_epoch%d.pt' % (fname, iepoch)))
            loss_eval = run_val(model, loss_obj, data_val, args)
            loss_evals.append(loss_eval/args.num_val)
    
        # save occupancy as figure
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
                iepoch, args, 'train')

        # save mesh as figure
        topology_vis = topology[:, :, loss_obj.visTopology]
        save_mesh_fig(
                pts_rnd[-1].data.cpu().numpy(),
                offset[-1],
                topology_vis[-1],
                loss_obj.x_grids,
                loss_obj.y_grids,
                loss_obj.z_grids,
                iepoch, args, 'train')

        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(np.arange(1, len(loss_epochs)+1), loss_epochs, 'k.-')
        axarr[1].plot(np.arange(1, len(loss_evals)+1)*args.snapshot, loss_evals, 'r.-')
        plt.savefig(os.path.join(args.output_dir, fname + '.png'))
        plt.close()

        np.savez(os.path.join(args.output_dir, fname),
                 loss_epochs=np.asarray(loss_epochs),
                 loss_evals=np.asarray(loss_evals))




if __name__ == '__main__':

    # parse args
    args = parse_args()

    # load data
    args, data_train = load_data(args, dtype, 'train')
    args, data_val = load_data(args, dtype, 'val')

    # setup loss object
    loss_obj = Loss(args)

    # initialize the model
    curr_epoch = 0
    if os.path.isfile(args.model):
        curr_epoch = int(os.path.splitext(args.model)[0][args.model.find('epoch')+len('epoch'):])
        print "Resuming training from epoch %d ..." % curr_epoch
        deep_marching_cubes = torch.load(args.model)
    else:
        deep_marching_cubes = DeepMarchingCube(args)

    if torch.cuda.is_available():
        deep_marching_cubes.cuda()

    # setup optimizer
    optimizer = torch.optim.Adam(deep_marching_cubes.parameters(),
                                 lr=args.learning_rate, 
                                 weight_decay=args.weight_decay)

    # train and validation
    run_train_val(deep_marching_cubes, optimizer, loss_obj, data_train, data_val, args)

    print 'Done!' 
