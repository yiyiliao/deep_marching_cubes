
int occupancy_connectivity_cuda_forward( THCudaTensor *occupancy, THCudaTensor *loss );
int occupancy_connectivity_cuda_backward( THCudaTensor *grad_output, THCudaTensor *occupancy, THCudaTensor *grad_occupancy );
