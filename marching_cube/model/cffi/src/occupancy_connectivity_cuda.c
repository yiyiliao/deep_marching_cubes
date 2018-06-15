#include <THC.h>
#include <THCGeneral.h>
#include <stdio.h>

#include "occupancy_connectivity_kernel.h"

extern THCState* state;

/*
 * Forward function, regularize the neighboring occupancy status to be close 
 * params: 
 *  	  occupancy 	input, (W+1)x(H+1)x(D+1)
 *  	  loss     	output, connectivity loss 
 *
 */	
int occupancy_connectivity_cuda_forward( THCudaTensor *occupancy, THCudaTensor *loss ){

  // data format check
  if (THCudaTensor_nDimension(state, occupancy)!=3 ||  THCudaTensor_nDimension(state, loss)!=1){
    printf("Invalid nDimension!\n");
    printf("Expected 3, 1, received %d, %d\n", THCudaTensor_nDimension(state, occupancy), THCudaTensor_nDimension(state, loss));
    return 0;
  }


  occupancy_connectivity_kernel_forward(state, occupancy, loss);

  return 1;

}

/*
 * Backward function, propagate the loss to every occupancy status 
 * params: 
 *  	  grad_output 		input, 1, gradient on the loss 
 *  	  occupancy 		input, (W+1)x(H+1)x(D+1)
 *  	  grad_occupancy     	output, (W+1)x(H+1)x(D+1), gradient on the occupancy 
 *
 */	
int occupancy_connectivity_cuda_backward( THCudaTensor *grad_output, THCudaTensor *occupancy, THCudaTensor *grad_occupancy ){

  occupancy_connectivity_kernel_backward(state, grad_output, occupancy, grad_occupancy);

  return 1;

}

