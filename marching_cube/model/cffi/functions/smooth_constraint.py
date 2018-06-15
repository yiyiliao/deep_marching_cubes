# functions/add.py
import torch
from torch.autograd import Function
from model.table import get_connected_pairs
from _ext import curvature, occupancy
if torch.cuda.is_available():
    from _ext import curvature_cuda, occupancy_cuda


x, y, z, inner, topology_to_triangles = get_connected_pairs()


class CurvatureFunction(Function):
    """ Compute the curvature loss
        see ../src/curvature_constraint.c
            ../src/curvature_constraint_cuda.c
            ../src/curvature_constraint_kernel.cu
        for more details
    """
    def forward(self, offset, topology):
        if not offset.is_cuda and not topology.is_cuda:
            loss = torch.zeros(1).type(torch.FloatTensor)
            curvature.curvature_constraint_forward(
			    offset,
			    topology[:, torch.LongTensor(topology_to_triangles)],
			    topology[:, -1],
			    torch.FloatTensor(x),
			    torch.FloatTensor(y),
			    torch.FloatTensor(z),
			    torch.FloatTensor(inner),
			    loss)
        else:
            loss = torch.zeros(1).type(torch.FloatTensor).cuda()
            curvature_cuda.curvature_constraint_cuda_forward(
			    offset,
			    topology[:, torch.LongTensor(topology_to_triangles).cuda()],
			    topology[:, -1],
			    torch.FloatTensor(x).cuda(),
			    torch.FloatTensor(y).cuda(),
			    torch.FloatTensor(z).cuda(),
			    torch.FloatTensor(inner).cuda(),
			    loss)
        self.save_for_backward(offset, topology)
        return loss

    def backward(self, grad_output):
        offset, topology = self.saved_tensors
        if not offset.is_cuda and not topology.is_cuda:
            grad_offset = torch.zeros(offset.size())
            curvature.curvature_constraint_backward(
			    grad_output,
			    offset,
			    topology[:, torch.LongTensor(topology_to_triangles)],
			    topology[:, -1],
			    torch.FloatTensor(x),
			    torch.FloatTensor(y),
			    torch.FloatTensor(z),
			    torch.FloatTensor(inner),
			    grad_offset)

            grad_topology = torch.zeros(topology.size())
        else:
            grad_offset = torch.zeros(offset.size()).cuda()
            curvature_cuda.curvature_constraint_cuda_backward(
			    grad_output,
			    offset,
			    topology[:, torch.LongTensor(topology_to_triangles).cuda()],
			    topology[:, -1],
			    torch.FloatTensor(x).cuda(),
			    torch.FloatTensor(y).cuda(),
			    torch.FloatTensor(z).cuda(),
			    torch.FloatTensor(inner).cuda(),
			    grad_offset)

            grad_topology = torch.zeros(topology.size()).cuda()
        #return grad_offset, None
        return grad_offset, grad_topology 


class OccupancyConnectivity(Function):
    """ Compute the smoothness loss between occupancy status
        see ../src/occupancy_connectivity.c
            ../src/occupancy_connectivity_cuda.c
            ../src/occupancy_connectivity_kernel.cu
        for more details
    """
    def forward(self, occ):
        if not occ.is_cuda:
            loss = torch.zeros(1).type(torch.FloatTensor)
            occupancy.occupancy_connectivity_forward(occ, loss)
        else:
            loss = torch.zeros(1).type(torch.FloatTensor).cuda()
            occupancy_cuda.occupancy_connectivity_cuda_forward(occ, loss)

        self.occ = occ
        return loss 

    def backward(self, grad_output):
        if not grad_output.is_cuda:
            grad_occupancy = torch.zeros(self.occ.size()).type(torch.FloatTensor)
            occupancy.occupancy_connectivity_backward(
			    grad_output,
			    self.occ,
			    grad_occupancy)
        else:
            grad_occupancy = torch.zeros(self.occ.size()).type(torch.FloatTensor).cuda()
            occupancy_cuda.occupancy_connectivity_cuda_backward(
			    grad_output,
			    self.occ,
			    grad_occupancy)
        # we only need gradient on feat_points
        return grad_occupancy 
