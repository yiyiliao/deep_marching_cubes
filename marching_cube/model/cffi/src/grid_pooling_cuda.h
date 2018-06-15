int grid_pooling_cuda_forward( THCudaTensor *point, THCudaTensor *feat_points, THLongTensor *shape, THCudaTensor *feat_cell, THCudaLongTensor *indices);
int grid_pooling_cuda_backward( THCudaTensor *grad_output, THLongTensor *shape, THCudaLongTensor *indices, THCudaTensor *grad_feat_points);
