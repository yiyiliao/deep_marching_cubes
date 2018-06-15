int point_topology_distance_cuda_forward(THCudaTensor *offset, THCudaTensor *points, THCudaTensor *distances, THCudaLongTensor *indices_all);
int point_topology_distance_cuda_backward(THCudaTensor *grad_output, THCudaTensor *offset, THCudaTensor *points, THCudaLongTensor *indices_all, THCudaTensor *grad_offset);
