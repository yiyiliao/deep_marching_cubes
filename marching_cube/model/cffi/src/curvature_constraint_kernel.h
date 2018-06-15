#ifdef __cplusplus
extern "C" {
#endif

void curvature_constraint_kernel_forward(THCState *state, THCudaTensor *offset, THCudaTensor *topology, THCudaTensor *xTable, THCudaTensor *yTable, THCudaTensor *zTable, THCudaTensor *innerTable, THCudaTensor *loss );
void curvature_constraint_kernel_backward(THCState *state, THCudaTensor *grad_output, THCudaTensor *offset, THCudaTensor *topology, THCudaTensor *xTable, THCudaTensor *yTable, THCudaTensor *zTable, THCudaTensor *innerTable, THCudaTensor *grad_offset );
	

#ifdef __cplusplus
}
#endif
