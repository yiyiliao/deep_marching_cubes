#include <THC.h>
#include <THCGeneral.h>
#include <stdio.h>

#include "occupancy_to_topology_kernel.h"

extern THCState* state;


/*
 * Forward function, compute the topology probability given the occupancy probability 
 * params: 
 *  	  occupancy 	input, (W+1)x(H+1)x(D+1)
 *  	  topology     	output, probability of all topologies types we care about (WxHxD)xT
 *
 */	
int occupancy_to_topology_cuda_forward( THCudaTensor *occupancy, THCudaTensor *topology ){

  int W = THCudaTensor_size(state, occupancy, 0)-1;
  int H = THCudaTensor_size(state, occupancy, 1)-1;
  int D = THCudaTensor_size(state, occupancy, 2)-1;

  // data format check
  if (THCudaTensor_nDimension(state, occupancy)!=3 ||  THCudaTensor_nDimension(state, topology)!=2){
    printf("Invalid nDimension!\n");
    printf("Expected 3, 2, received %d, %d\n", THCudaTensor_nDimension(state, occupancy), THCudaTensor_nDimension(state, topology));
    return 0;
  }
  if (THCudaTensor_size(state, topology,0)!=W*H*D){
    printf("Invalid shape of topology!\n");
    return 0;
  }

  occupancy_to_topology_kernel_forward(state, occupancy, topology);

  return 1;
}



/*
 * Backward function, backpropagate the gradient from topology to occupancy 
 * params: 
 * 	  grad_output   	input, gradient on the topology probability, (WxHxD)xT
 *  	  occupancy 		input, (W+1)x(H+1)x(D+1)
 *  	  topology     		input, probability of all topologies types we care about (WxHxD)xT
 *  	  grad_occupancy   	output, gradient on the occupancy map, (W+1)x(H+1)x(D+1) 
 *
 */	
int occupancy_to_topology_cuda_backward( THCudaTensor *grad_output, THCudaTensor *occupancy, THCudaTensor *topology, THCudaTensor *grad_occupancy ){

  occupancy_to_topology_kernel_backward(state, grad_output, occupancy, topology, grad_occupancy);

  return 1;
}
