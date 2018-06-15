# functions/point_triangle_distance.py
import torch
from torch.autograd import Function
from _ext import distance 
if torch.cuda.is_available():
    from _ext import distance_cuda


class DistanceFunction(Function):
    """ Calculate the distance between the points and the triangles 
        see ../src/point_triangle_distance.c
            ../src/point_triangle_distance_cuda.c
            ../src/point_triangle_distance_kernel.cu
        for more details
    """
    def forward(self, offset, points):
        W = offset.size()[1]
        H = offset.size()[2]
        D = offset.size()[3]

        # we only considered topologies with up to 3 triangles for calculating
        # the distance loss function, the distance can be calculated in regardless
        # of the normal vectors, therefore there are only 48 topologies to be
        # considered
        T = 48

        if not offset.is_cuda and not points.is_cuda:
            distances_full = torch.zeros((W-1)*(H-1)*(D-1), T)
            indices = -1 * torch.ones(points.size(0), T).type(torch.LongTensor)
            distance.point_topology_distance_forward(
                    offset, points, distances_full, indices)
        else:
            distances_full = torch.zeros((W-1)*(H-1)*(D-1), T).cuda()
            indices = -1 * torch.ones(points.size(0), T).type(torch.LongTensor).cuda()
            distance_cuda.point_topology_distance_cuda_forward(
                    offset, points, distances_full, indices) 
        self.save_for_backward(offset, points)
        self.indices_all = indices
        return distances_full 

    def backward(self, grad_output):
        offset, points = self.saved_tensors

        if not offset.is_cuda and not points.is_cuda:
            grad_offset = torch.zeros(offset.size())
            distance.point_topology_distance_backward(
                    grad_output, offset, points, self.indices_all, grad_offset)
        else:
            grad_offset = torch.zeros(offset.size()).cuda()
            distance_cuda.point_topology_distance_cuda_backward(
                    grad_output, offset, points, self.indices_all, grad_offset)
        return grad_offset, None 
