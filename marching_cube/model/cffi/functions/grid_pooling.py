# functions/add.py
import torch
from torch.autograd import Function
from _ext import forward_utils 
if torch.cuda.is_available():
    from _ext import forward_utils_cuda

class GridPooling(Function):
    """ Perform max-pooling in every cell over the point features
        see ../src/grid_pooling.c
            ../src/grid_pooling_cuda.c
            ../src/grid_pooling_kernel.cu
        for more details
    """
    def forward(self, feat_points, points, grid_shape):
        W = grid_shape[0]
        H = grid_shape[1]
        D = grid_shape[2]
        C = feat_points.size()[1]
        if not feat_points.is_cuda and not points.is_cuda:
            feat_cells = torch.zeros(W*H*D, C).type(torch.FloatTensor)
            indices = -1 * torch.ones(W*H*D, C).type(torch.LongTensor)
            forward_utils.grid_pooling_forward(points, feat_points, grid_shape, feat_cells, indices)
        else:
            feat_cells = torch.zeros(W*H*D, C).type(torch.FloatTensor).cuda()
            indices = -1 * torch.ones(W*H*D, C).type(torch.LongTensor).cuda()
            forward_utils_cuda.grid_pooling_cuda_forward(points, feat_points, grid_shape, feat_cells, indices)

        # save max indices for back-propagation
        self.saved_indices = indices
        # save number of points and feature dimension for back-propagation
        self.N = points.size()[0]
        self.C = C
        self.grid_shape = grid_shape
        return feat_cells

    def backward(self, grad_output):
        if not grad_output.is_cuda:
            grad_points = torch.zeros(self.N, self.C).type(torch.FloatTensor)
            forward_utils.grid_pooling_backward(grad_output, self.saved_indices, grad_points)
        else:
            grad_points = torch.zeros(self.N, self.C).type(torch.FloatTensor).cuda()
            forward_utils_cuda.grid_pooling_cuda_backward(grad_output, self.grid_shape, self.saved_indices, grad_points)
        # we only need gradient on feat_points
        return grad_points, None, None
