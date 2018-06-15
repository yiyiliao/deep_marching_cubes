import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import time
import numpy as np

from loss import Loss 
from loss_autograd import LossAutoGrad 
from parse_args import parse_args

# check the cuda extension or c extension
args = parse_args()
if args.with_cuda:
    print "Testing CUDA extension..."
    dtype = torch.cuda.FloatTensor
else:
    print "Testing C extension..."
    dtype = torch.FloatTensor


# cffi loss
loss_cffi = Loss(args)

# auto loss
loss_auto = LossAutoGrad(args)


if __name__ == '__main__':

    W = H = D = args.num_cells
    occupancy = Variable(torch.rand(W+1, H+1, D+1).type(dtype), requires_grad=True)
    rnd_weights = Variable(torch.rand(W*H*D, 48).type(dtype))

    print "=========== Input ============="
    print occupancy

    print "============= cffi ============"
    # forward
    loss = 0.1*loss_cffi.loss_on_smoothness(occupancy)*args.num_cells**3
    tf_c = time.time()
    loss = 0.1*loss_cffi.loss_on_smoothness(occupancy)*args.num_cells**3
    tf_c = time.time() - tf_c
    print "cffi forward time: ", tf_c
    print loss

    # backward
    tb_c = time.time()
    loss.backward()
    tb_c = time.time() - tb_c
    print "cffi backward time: ", tb_c

    grad_np = np.copy(occupancy.grad.data.cpu().numpy())
    print grad_np

    print "============= auto ============"
    occupancy = Variable(occupancy.data.cpu(), requires_grad=True)
    rnd_weights = Variable(rnd_weights.data.cpu())

    # forward
    tf_py = time.time()
    loss_auto = 0.1*loss_auto.loss_on_smoothness_autograd(occupancy)
    tf_py = time.time()-tf_py
    print "auto forward time: ", tf_py
    print loss_auto

    # backward
    #occupancy.grad.data.zero_()
    tb_py = time.time()
    loss_auto.backward()
    tb_py = time.time()-tb_py
    print "auto backward time: ", tf_py

    grad_auto_np = np.copy(occupancy.grad.data.cpu().numpy())
    print grad_auto_np

    print "========== summary ==========="
    print "Forward difference between cffi and auto: ", np.sum(np.abs(loss.data.cpu().numpy()-loss_auto.data.cpu().numpy()))
    print "Backward difference between cffi and auto: ", np.sum(np.abs(grad_np-grad_auto_np))

    print "cffi forward time: %f, backward time: %f, full time: %f " % (tf_c, tb_c, tf_c+tb_c)
    print "auto forward time: %f, backward time: %f, full time: %f " % (tf_py, tb_py, tf_py+tb_py)
    print "ratio: ", (tf_py+tb_py)/(tf_c + tb_c)
