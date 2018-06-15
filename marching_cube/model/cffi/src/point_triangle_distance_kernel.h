#ifdef __cplusplus
extern "C" {
#endif

void point_topology_distance_kernel_forward(THCState *state, THCudaTensor *offset, THCudaTensor *points, THCudaTensor *distances, THCudaLongTensor *indices_all);
void point_topology_distance_kernel_backward(THCState *state, THCudaTensor *grad_output, THCudaTensor *offset, THCudaTensor *points, THCudaLongTensor *indices_all, THCudaTensor *grad_offset);

#ifdef __cplusplus
}
#endif
