#include <THC.h>
#include <THCGeneral.h>
#include <stdio.h>
#include "point_triangle_distance_kernel.h"

extern THCState* state;

/*
 * Forward function, calculating the point to mesh distances for all grids
 * params: 
 * 	  offset 	input, vertex displacement field, 3x(W+1)x(H+1)x(D+1) 
 *  	  points 	input, all points, N_allx3
 *  	  distances  	output, point to mesh distances for every grid for every topolopy, (WxHxD)xT 
 *  	  indices_all   output, to record which triangle in each topology is the nearest one for backpropagation, N_allxT
 *
 */	
int point_topology_distance_cuda_forward(THCudaTensor *offset, THCudaTensor *points, THCudaTensor *distances, THCudaLongTensor *indices_all){
  // data format check
  if (THCudaTensor_nDimension(state, offset)!=4 ||  THCudaTensor_nDimension(state, points)!=2 || THCudaTensor_nDimension(state, distances)!=2){
    printf("Invalid nDimension!\n");
    printf("Expected 4, 2, 2, received %d, %d, %d \n", THCudaTensor_nDimension(state, offset), THCudaTensor_nDimension(state, points), THCudaTensor_nDimension(state, distances));
    return 0;
  }

  point_topology_distance_kernel_forward(state, offset, points, distances, indices_all);

  return 1;
  
}

/*
 * Backward function, calculating the gradients for the full offset map 
 * params: 
 *  	  grad_output   input, gradient on the output distances, (WxHxD)xT
 * 	  offset 	input, vertex displacement field, 3x(W+1)x(H+1)x(D+1) 
 *  	  points 	input, all points, N_allx3
 *  	  indices_all   input, recorded which triangle in each topology is the nearest one for backpropagation, N_allxT
 *  	  grad_offset  	output, gradient on the full offset map, 3x(W+1)x(H+1)x(D+1)  
 *
 */	
int point_topology_distance_cuda_backward(THCudaTensor *grad_output, THCudaTensor *offset, THCudaTensor *points, THCudaLongTensor *indices_all, THCudaTensor *grad_offset){

  point_topology_distance_kernel_backward(state, grad_output, offset, points, indices_all, grad_offset);

  return 1;
}
