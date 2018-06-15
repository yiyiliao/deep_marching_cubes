# functions/add.py
import torch
from torch.autograd import Function
from _ext import forward_utils 
if torch.cuda.is_available():
    from _ext import forward_utils_cuda 


class OccupancyToTopology(Function):
    """ Convert the occupancy probability to topology probability
        see ../src/occupancy_to_topology.c 
            ../src/occupancy_connectivity_cuda.c
            ../src/occupancy_to_topology_kernel.cu 
        for more details
    """
    def forward(self, occupancy):
        W = occupancy.size()[0]-1
        H = occupancy.size()[1]-1
        D = occupancy.size()[2]-1

        T = 256
        if not occupancy.is_cuda:
            topology = torch.zeros(W*H*D, T).type(torch.FloatTensor)
            forward_utils.occupancy_to_topology_forward(occupancy, topology)
        else:
            topology = torch.zeros(W*H*D, T).type(torch.FloatTensor).cuda()
            forward_utils_cuda.occupancy_to_topology_cuda_forward(occupancy, topology)

        self.occupancy = occupancy
        self.topology = topology 

        return topology 

    def backward(self, grad_output):
        if not grad_output.is_cuda:
            grad_occupancy = torch.zeros(self.occupancy.size()).type(torch.FloatTensor)
            forward_utils.occupancy_to_topology_backward(grad_output, self.occupancy, self.topology, grad_occupancy)
        else:
            grad_occupancy = torch.zeros(self.occupancy.size()).type(torch.FloatTensor).cuda()
            forward_utils_cuda.occupancy_to_topology_cuda_backward(grad_output, self.occupancy, self.topology, grad_occupancy)
        # we only need gradient on feat_points
        return grad_occupancy 
