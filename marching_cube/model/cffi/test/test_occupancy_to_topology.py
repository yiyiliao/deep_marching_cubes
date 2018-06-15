import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from table import get_occupancy_table
from functions.occupancy_to_topology import OccupancyToTopology 
from parse_args import parse_args

# look-up-tables
acceptTopology = np.arange(256)
vertexTable=[ [0, 1, 0],
	      [1, 1, 0],
	      [1, 0, 0],
              [0, 0, 0],
	      [0, 1, 1],
	      [1, 1, 1],
	      [1, 0, 1],
              [0, 0, 1] ]
occupancyTable=get_occupancy_table()

# check the cuda extension or c extension
args = parse_args()
if args.with_cuda:
    print "Testing CUDA extension..."
    dtype = torch.cuda.FloatTensor
else:
    print "Testing C extension..."
    dtype = torch.FloatTensor


# get (WxH)xT topology map from (W+1)x(Hx1) occupancy map
# note here T=14 because of the inside/outside distinction
def occupancy_to_topology(occ):
    W = occ.size()[0]-1
    H = occ.size()[1]-1
    D = occ.size()[2]-1
    T = len(acceptTopology)
    topology = Variable(torch.zeros(W*H*D, T)).type(torch.FloatTensor)

    xv, yv, zv = np.meshgrid(range(W), range(H), range(D), indexing='ij')
    xv = xv.flatten()
    yv = yv.flatten()
    zv = zv.flatten()
    
    for i,j,k in zip(xv, yv, zv):
        p_occ = [] 
        for v in range(8):
            p_occ.append( occ[i+vertexTable[v][0], j+vertexTable[v][1], k+vertexTable[v][2]] )
            p_occ.append( 1 - occ[i+vertexTable[v][0], j+vertexTable[v][1], k+vertexTable[v][2]] )
        for t in range(T):
            topology_ind = acceptTopology[t]
            p_accumu = 1
            for v in range(8):
                p_accumu = p_accumu*p_occ[ v*2 + int(occupancyTable[topology_ind][v]) ] 
            topology[i*H*D+j*D+k, t] = p_accumu
    return topology



if __name__ == '__main__':

    W = H = D = args.num_cells
    T = 256

    print "=========== Input ============="
    occupancy = Variable(torch.rand(W+1, H+1, D+1).type(dtype), requires_grad=True)
    rnd_weights = Variable(torch.rand(W*H*D, T).type(dtype))
    print occupancy

    print "============= cffi ============"
    # forward
    topology = OccupancyToTopology()(occupancy)

    tf_c = time.time()
    topology = OccupancyToTopology()(occupancy)
    tf_c = time.time() - tf_c
    print "cffi forward time: ", tf_c
    print topology

    # backward
    tb_c = time.time()
    torch.sum(torch.mul(topology, rnd_weights)).backward()
    tb_c = time.time() - tb_c
    print "cffi backward time: ", tb_c

    grad_np = np.copy(occupancy.grad.data.cpu().numpy())
    print grad_np

    print "============= auto ============"
    occupancy = Variable(occupancy.data.cpu(), requires_grad=True)
    rnd_weights = Variable(rnd_weights.data.cpu())

    # forward
    tf_py = time.time()
    topology_auto = occupancy_to_topology(occupancy)
    tf_py = time.time()-tf_py
    print "auto forward time: ", tf_py
    print topology_auto

    # backward
    #occupancy.grad.data.zero_()
    tb_py = time.time()
    torch.sum(torch.mul(topology_auto, rnd_weights)).backward()
    tb_py = time.time()-tb_py
    print "auto backward time: ", tf_py

    grad_auto_np = np.copy(occupancy.grad.data.cpu().numpy())
    print grad_auto_np

    print "========== summary ==========="
    print "Forward difference between cffi and auto: ", np.sum(np.abs(topology.data.cpu().numpy()-topology_auto.data.numpy()))
    print "Backward difference between cffi and auto: ", np.sum(np.abs(grad_np-grad_auto_np))

    print "cffi forward time: %f, backward time: %f, full time: %f " % (tf_c, tb_c, tf_c+tb_c)
    print "auto forward time: %f, backward time: %f, full time: %f " % (tf_py, tb_py, tf_py+tb_py)
    print "ratio: ", (tf_py+tb_py)/(tf_c + tb_c)
    
