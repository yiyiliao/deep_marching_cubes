#ifdef __cplusplus
extern "C" {
#endif

void occupancy_to_topology_kernel_forward( THCState *state, THCudaTensor *occupancy, THCudaTensor *topology );
void occupancy_to_topology_kernel_backward( THCState *state, THCudaTensor *grad_output, THCudaTensor *occupancy, THCudaTensor *topology, THCudaTensor *grad_occupancy );

#ifdef __cplusplus
}
#endif
