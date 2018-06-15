import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
from utils.util import gaussian_kernel, offset_to_normal 
from model.cffi.modules.point_triangle_distance import DistanceModule
from model.cffi.modules.smooth_constraint import CurvatureModule, OccupancyModule
from model.table import get_accept_topology

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor

one = Variable(torch.ones(1).type(dtype), requires_grad=True)
eps = 1e-6
dis_empty = Variable(torch.ones(48).type(dtype))
dis_empty[-1].data[0] = 0.0

class Loss(object):
    """Compute the losses given the predicted mesh """

    def __init__(self, args):
        self.x_grids = np.arange(0, args.num_cells+1, args.len_cell)
        self.y_grids = np.arange(0, args.num_cells+1, args.len_cell)
        self.z_grids = np.arange(0, args.num_cells+1, args.len_cell)

        self.args = args
        self.distanceModule = DistanceModule()
        self.curvatureLoss = CurvatureModule()
        self.occupancyConnectivity = OccupancyModule()

        self.acceptTopology = torch.LongTensor(get_accept_topology())
 	if torch.cuda.is_available():
            self.acceptTopology = self.acceptTopology.cuda()
        flip_indices = torch.arange(self.acceptTopology.size()[0]-1, -1, -1).type(dtype_long)
        self.acceptTopologyWithFlip = torch.cat([self.acceptTopology, 255-self.acceptTopology[flip_indices]], dim=0)

        # TODO: consider the topology with 4 triangles only for visualizing, need to be fixed
        self.visTopology = torch.LongTensor(get_accept_topology(4))
 	if torch.cuda.is_available():
            self.visTopology = self.visTopology.cuda()

        # assume that the outside __faces__ of the grid is always free
        W = len(self.x_grids)
        H = len(self.y_grids)
        D = len(self.z_grids)

        tmp_ = np.zeros((W,H,D))
        tmp_[0,     :,      :] = 1
        tmp_[W-1,   :,      :] = 1
        tmp_[:,     :,      0] = 1
        tmp_[:,     :,    D-1] = 1
        tmp_[:,     0,      :] = 1
        tmp_[:,     H-1,    :] = 1
        kern3 = gaussian_kernel(3) 
        neg_weight = scipy.ndimage.filters.convolve(tmp_, kern3)
        neg_weight = neg_weight/np.max(neg_weight)
        self.neg_weight = Variable(torch.from_numpy(neg_weight).type(dtype))


    def loss_point_to_mesh(self, offset, topology, pts, phase='train'):
        """Compute the point to mesh distance"""

        # compute the distances between all topologies and a point set
        dis_sub = self.distanceModule(offset, pts)

        # dual topologies share the same point-to-triangle distance
        flip_indices = torch.arange(len(self.acceptTopology)-1, -1, -1).type(dtype_long)
        dis_accepted = torch.cat([dis_sub, dis_sub[:, flip_indices]], dim=1)
        topology_accepted = topology[:, self.acceptTopologyWithFlip]

        # renormalize all desired topologies so that they sum to 1
        # TODO: add clamp as the sum might be zero, not sure if it will cause unstability 
        prob_sum = torch.sum(topology_accepted, dim=1, keepdim=True).clamp(1e-6)
        topology_accepted = topology_accepted / prob_sum

        # compute the expected loss
        loss = torch.sum(topology_accepted.mul(dis_accepted)) /  (self.args.num_cells**3)

        if phase == 'train':
            loss = loss * self.args.weight_distance

        return loss


    def loss_on_occupancy(self, occupancy):
        """Compute the loss given the prior that the 6 faces of the cube 
        bounding the 3D scene are unoccupied and a sub-volume inside the
        scene is occupied
        """

        # loss on 6 faces of the cube
        loss_free = torch.sum(torch.mul(occupancy, self.neg_weight)) \
                /torch.sum(self.neg_weight)

        W=occupancy.size()[0]
        H=occupancy.size()[1]
        D=occupancy.size()[2]

        # get occupancy.data as we don't want to backpropagate to the adaptive_weight
        sorted_cube,_ = torch.sort(occupancy.data.view(-1), 0, descending=True)
        # check the largest 1/30 value
        adaptive_weight = 1 - torch.mean(sorted_cube[0:int(sorted_cube.size()[0]/30)])

        # loss on a subvolume inside the cube, where the weight is assigned
        # adaptively w.r.t. the current occupancy status 
        loss_occupied = self.args.weight_prior_pos * adaptive_weight * \
                (1-torch.mean(occupancy[int(0.2*W):int(0.8*W), \
                int(0.2*H):int(0.8*H), int(0.2*D):int(0.8*D)]))

        return (loss_free + loss_occupied) * self.args.weight_prior


    def loss_on_smoothness(self, occupancy):
        """Compute the smoothness loss defined between neighboring occupancy
        variables
        """
        return self.occupancyConnectivity(occupancy) / (self.args.num_cells**3) \
                * self.args.weight_smoothness

    def loss_on_curvature(self, offset, topology):
        """Compute the curvature loss by measuring the smoothness of the
        predicted mesh geometry
        """
        topology_accepted = topology[:, self.acceptTopologyWithFlip]
        return self.args.weight_curvature*self.curvatureLoss(offset, \
                F.softmax(topology_accepted, dim=1)) / (self.args.num_cells**3)

    def loss_train(self, offset, topology, pts, occupancy):
        """Compute the losses given a batch of point cloud and the predicted
        mesh during the training phase
        """
        loss = 0
        loss_stages = []

        batchsize = offset.size()[0]

        for i in range(batchsize):

            # L^{mesh}
            loss += self.loss_point_to_mesh(offset[i], topology[i], pts[i], 'train')
            if i == 0:
                loss_stages.append(loss.data[0])

            # L^{occ}
            loss += self.loss_on_occupancy(occupancy[i, 0])
            if i == 0:
                loss_stages.append(loss.data[0] - sum(loss_stages))

            # L^{smooth}
            loss += self.loss_on_smoothness(occupancy[i, 0])
            if i == 0:
                loss_stages.append(loss.data[0] - sum(loss_stages))

            # L^{curve}
            loss += self.loss_on_curvature(offset[i], topology[i])
            if i == 0:
                loss_stages.append(loss.data[0] - sum(loss_stages))


        loss = loss/batchsize

        return loss, loss_stages


    def loss_eval(self, offset, topology, pts):
        """Compute the point to mesh loss during validation phase"""
        loss = self.loss_point_to_mesh(offset, topology, pts, 'val')
        return loss*one

