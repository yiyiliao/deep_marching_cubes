int occupancy_to_topology_cuda_forward( THCudaTensor *occupancy, THCudaTensor *topology );
int occupancy_to_topology_cuda_backward( THCudaTensor *grad_output, THCudaTensor *occupancy, THCudaTensor *topology, THCudaTensor *grad_occupancy );
