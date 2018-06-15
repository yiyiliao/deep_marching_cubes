#ifdef __cplusplus
extern "C" {
#endif

void grid_pooling_kernel_forward( THCState *state, THCudaTensor *point, THCudaTensor *feat_points, THLongTensor *shape, THCudaTensor *feat_cell, THCudaLongTensor *indices);
void grid_pooling_kernel_backward( THCState *state, THCudaTensor *grad_output, THLongTensor *shape, THCudaLongTensor *indices, THCudaTensor *grad_feat_points);

#ifdef __cplusplus
}
#endif
