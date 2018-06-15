from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from functions.grid_pooling import GridPooling
from functions.occupancy_to_topology import OccupancyToTopology 

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

def point_to_cell(pts, feat, W, H, D, expand=1):
    """ perform maxpool on points in every cell
    set zero vector if cell is empty
    if expand=1 then return (N+1)x(N+1)x(N+1), for dmc
       expand=0 then return NxNxN, for occupancy/sdf baselines
    """
    batchsize = feat.size()[0]
    C = feat.size()[2] 
    feat_cell = Variable(torch.zeros(batchsize, W*H*D, C).type(dtype))
    grid_shape = Variable(torch.LongTensor([W, H, D]))
    for k in range(batchsize):
        feat_cell[k, :, :] = GridPooling()(feat[k, :, :], pts[k, :, :], grid_shape) 

    feat_cell = torch.transpose(feat_cell, 1, 2).contiguous().view(-1, C, W, H, D)
    if expand == 0:
        return feat_cell

    # expand to (W+1)x(H+1)
    curr_size = feat_cell.size()
    feat_cell_exp = Variable(torch.zeros(curr_size[0], curr_size[1], curr_size[2]+1, curr_size[3]+1, curr_size[4]+1).type(dtype))
    feat_cell_exp[:, :, :-1, :-1, :-1] = feat_cell
    return feat_cell_exp

class PointNetfeatLocal(nn.Module):
    """Learn point-wise feature in the beginning of the network
    with fully connected layers the same as PointNet, the fully
    connected layers are implemented as 1d convolution so that
    it is independent to the number of points
    """ 
    def __init__(self):
        super(PointNetfeatLocal, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 16, 1)
    def forward(self, x):
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        pointfeat = x.transpose(1, 2)
        return pointfeat
        
class LocalEncoder(nn.Module):
    """Encoder of the U-Net"""
    def __init__(self, input_dim=16, skip_connection=True):
        super(LocalEncoder, self).__init__()

        ## u-net
        self.conv1_1 = nn.Conv3d(input_dim, 16, 3, padding=3)
        self.conv1_2 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv2_1 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv2_2 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv3_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv3d(64, 128, 3, padding=1)

	# batchnorm
        self.conv1_1_bn = nn.BatchNorm3d(16)
        self.conv1_2_bn = nn.BatchNorm3d(16)
        self.conv2_1_bn = nn.BatchNorm3d(32)
        self.conv2_2_bn = nn.BatchNorm3d(32)
        self.conv3_1_bn = nn.BatchNorm3d(64)
        self.conv3_2_bn = nn.BatchNorm3d(64)
        self.conv4_bn   = nn.BatchNorm3d(128)

        self.maxpool = nn.MaxPool3d(2, return_indices=True)

        self.skip_connection = skip_connection

    def encoder(self, x):
	#
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        feat1 = x
        size1 = x.size()
        x, indices1 = self.maxpool(x)

        #
        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        feat2 = x
        size2 = x.size()
        x, indices2 = self.maxpool(x)

        #
        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
        feat3 = x
        size3 = x.size()
        x, indices3 = self.maxpool(x)

        #
        x = F.relu(self.conv4_bn(self.conv4(x)))
        return x, feat1, size1, indices1, feat2, size2, indices2, feat3, size3, indices3

    def forward(self, x):
        x, feat1, size1, indices1, feat2, size2, indices2, feat3, size3, indices3 = self.encoder(x)
        if self.skip_connection: 
            return x, (feat1, size1, indices1, feat2, size2, indices2, feat3, size3, indices3)
        else:
            return x


class SurfaceDecoder(nn.Module):
    """Decoder of the U-Net, estimate topology and offset with two headers"""
    def __init__(self, skip_connection=True):
        super(SurfaceDecoder, self).__init__()

	# decoder
        self.deconv4 = nn.Conv3d(128, 64, 3, padding=1)
        self.deconv3_1 = nn.ConvTranspose3d(128, 128, 3, padding=1)
        self.deconv3_2 = nn.ConvTranspose3d(128, 32, 3, padding=1)
        self.deconv2_off_1 = nn.ConvTranspose3d(64, 64, 3, padding=1)
        self.deconv2_off_2 = nn.ConvTranspose3d(64, 16, 3, padding=1)
        self.deconv2_occ_1 = nn.ConvTranspose3d(64, 64, 3, padding=1)
        self.deconv2_occ_2 = nn.ConvTranspose3d(64, 16, 3, padding=1)
        self.deconv1_off_1 = nn.ConvTranspose3d(32, 32, 3, padding=1)
        self.deconv1_off_2 = nn.ConvTranspose3d(32, 3 , 3, padding=3)
        self.deconv1_occ_1 = nn.ConvTranspose3d(32, 32, 3, padding=1)
        self.deconv1_occ_2 = nn.ConvTranspose3d(32, 1 , 3, padding=3)
        
        # batchnorm
        self.deconv4_bn = nn.BatchNorm3d(64)
        self.deconv3_1_bn = nn.BatchNorm3d(128)
        self.deconv3_2_bn = nn.BatchNorm3d(32)
        self.deconv2_off_1_bn = nn.BatchNorm3d(64)
        self.deconv2_off_2_bn = nn.BatchNorm3d(16)
        self.deconv2_occ_1_bn = nn.BatchNorm3d(64)
        self.deconv2_occ_2_bn = nn.BatchNorm3d(16)
        self.deconv1_off_1_bn = nn.BatchNorm3d(32)
        self.deconv1_occ_1_bn = nn.BatchNorm3d(32)

        self.sigmoid = nn.Sigmoid()

        self.maxunpool = nn.MaxUnpool3d(2)

        self.skip_connection = skip_connection

    def decoder(self, x, intermediate_feat=None):

        if self.skip_connection:
            feat1, size1, indices1, feat2, size2, indices2, feat3, size3, indices3 = intermediate_feat

	#
        x = F.relu(self.deconv4_bn(self.deconv4(x)))

	#
        x = self.maxunpool(x, indices3, output_size=size3)
        if self.skip_connection:
            x = torch.cat((feat3, x), 1)
        x = F.relu(self.deconv3_1_bn(self.deconv3_1(x)))
        x = F.relu(self.deconv3_2_bn(self.deconv3_2(x)))

        #
        x = self.maxunpool(x, indices2, output_size=size2)
        if self.skip_connection:
            x = torch.cat((feat2, x), 1)
        x_occupancy = F.relu(self.deconv2_occ_1_bn(self.deconv2_occ_1(x)))
        x_occupancy = F.relu(self.deconv2_occ_2_bn(self.deconv2_occ_2(x_occupancy)))
        x_offset = F.relu(self.deconv2_off_1_bn(self.deconv2_off_1(x)))
        x_offset = F.relu(self.deconv2_off_2_bn(self.deconv2_off_2(x_offset)))

        #
        x_occupancy = self.maxunpool(x_occupancy, indices1, output_size=size1)
        if self.skip_connection:
            x_occupancy = torch.cat((feat1, x_occupancy), 1)
        x_offset = self.maxunpool(x_offset, indices1, output_size=size1)
        if self.skip_connection:
            x_offset = torch.cat((feat1, x_offset), 1)
        x_occupancy = F.relu(self.deconv1_occ_1_bn(self.deconv1_occ_1(x_occupancy)))
        x_occupancy = self.sigmoid(self.deconv1_occ_2(x_occupancy))
        x_offset = F.relu(self.deconv1_off_1_bn(self.deconv1_off_1(x_offset)))
        x_offset = self.sigmoid(self.deconv1_off_2(x_offset)) - 0.5

        return x_occupancy, x_offset

    def forward(self, x, intermediate_feat=None):
        return self.decoder(x, intermediate_feat)


class DeepMarchingCube(nn.Module):
    """Network architecture of Deep Marching Cubes"""
    def __init__(self, args):
        super(DeepMarchingCube, self).__init__()
        self.input_data = args.encoder_type

        self.feat = PointNetfeatLocal()

        self.skip_connection = args.skip_connection
        if self.input_data == 'point':
            self.encoder = LocalEncoder(16, self.skip_connection)
            self.decoder = SurfaceDecoder(self.skip_connection)
        elif self.input_data == 'voxel':
            self.encoder = LocalEncoder(1, self.skip_connection)
            self.decoder = SurfaceDecoder(self.skip_connection)

        self.W = args.num_cells
        self.H = args.num_cells
        self.D = args.num_cells

    def forward(self, x):
        if self.input_data == 'point':
            z = self.feat(x)
            x = point_to_cell(x, z, self.W, self.H, self.D)
        elif self.input_data == 'voxel':
            x = x

        if self.skip_connection:
            x, intermediate_feat = self.encoder(x)
            occupancy, offset = self.decoder(x, intermediate_feat)
        else:
            x = self.encoder(x)
            occupancy, offset = self.decoder(x)

        batchsize = occupancy.size()[0]

        T = 256
        topology = Variable(torch.zeros(batchsize, self.W*self.H*self.D, T).type(dtype))
        for k in range(batchsize):
            topology[k, :, :] = OccupancyToTopology()(occupancy[k, 0, :, :])

        return offset, topology, occupancy 

