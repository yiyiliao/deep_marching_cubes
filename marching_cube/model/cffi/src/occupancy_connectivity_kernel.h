#ifdef __cplusplus
extern "C" {
#endif

void occupancy_connectivity_kernel_forward( THCState *state, THCudaTensor *occupancy, THCudaTensor *loss );
void occupancy_connectivity_kernel_backward( THCState *state, THCudaTensor *grad_output, THCudaTensor *occupancy, THCudaTensor *grad_occupancy );

#ifdef __cplusplus
}
#endif
