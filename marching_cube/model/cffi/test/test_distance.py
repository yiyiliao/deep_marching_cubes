import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import numpy as np
import resource
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from modules.point_triangle_distance import DistanceModule 
#from utils.pointTriangleDistance import pointTriangleDistanceFast
#from utils.util import pts_in_cell, offset_to_vertices, dis_to_meshs
from loss_autograd import LossAutoGrad
from parse_args import parse_args

# check the cuda extension or c extension
args = parse_args()
if args.with_cuda:
    print "Testing CUDA extension..."
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    print "Testing C extension..."
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor

# autograd loss
loss_autograd = LossAutoGrad(args)

# cffi loss
class MultipleGrids(nn.Module):
    def __init__(self):
        super(MultipleGrids, self).__init__()
        self.distance = DistanceModule()

    def forward(self, input1, input2):
        return self.distance(input1, input2)

multiGrids = MultipleGrids()


if __name__ == '__main__':

    print "=========== Input ============="
    #point = Variable(torch.rand(10, 3).view(-1,3).type(dtype)) * args.num_cells 
    #offset = Variable(torch.rand(3, args.num_cells+1, args.num_cells+1, args.num_cells+1).type(dtype)*0.5, requires_grad=True)
    point = Variable(torch.ones(1, 3).view(-1,3).type(dtype) * 0.9) * args.num_cells 
    offset = Variable(torch.zeros(3, args.num_cells+1, args.num_cells+1, args.num_cells+1).type(dtype)*0.5, requires_grad=True)
    print point
    print offset
    
    print "============= cffi ============"
    # forward
    tf_c = time.time()
    distance = multiGrids(offset, point)
    tf_c = time.time() - tf_c
    distance_np = distance.data.cpu().numpy()
    print "cffi distance:"
    print distance_np

    weight_rnd = Variable(torch.rand(distance.size()).type(dtype), requires_grad=False)
    distance_sum = torch.sum(torch.mul(distance, weight_rnd))
    #distance_sum = torch.sum(distance)
    
    # backward
    tb_c = time.time()
    distance_sum.backward()
    tb_c = time.time() - tb_c
    offset_np = np.copy(offset.grad.data.cpu().numpy())
    
    print "cffi grad:"
    print offset_np
    
    print "============= auto ============"
    # forward
    tf_py = time.time()
    distance_auto = loss_autograd.loss_point_to_mesh_distance_autograd(offset, point)
    tf_py = time.time()-tf_py
    distance_auto_np = distance_auto.data.cpu().numpy()
    print "auto distance:"
    print distance_auto_np
    weight_rnd = Variable(weight_rnd.data)
    distance_sum_auto = torch.sum(torch.mul(distance_auto, weight_rnd))

    # backward
    offset.grad.data.zero_()
    
    tb_py = time.time()
    distance_sum_auto.backward()
    tb_py = time.time() - tb_py
    print "auto grad: "
    offset_auto_np = np.copy(offset.grad.data.cpu().numpy())
    print offset_auto_np

    print "========== summary ==========="
    print "Forward difference between cffi and auto: ", np.sum(np.abs(distance_np[:,:-1]-distance_auto_np[:,:-1]))
    print "Backward difference between cffi and auto: ", np.sum(np.abs(offset_np-offset_auto_np))
    
    print "cffi forward time: %f, backward time: %f, full time: %f " % (tf_c, tb_c, tf_c+tb_c)
    print "auto forward time: %f, backward time: %f, full time: %f " % (tf_py, tb_py, tf_py+tb_py)
    print "ratio: ", (tf_py+tb_py)/(tf_c + tb_c)



