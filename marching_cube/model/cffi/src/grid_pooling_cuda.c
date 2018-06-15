#include <THC.h>
#include <THCGeneral.h>
#include <stdio.h>

#include "grid_pooling_kernel.h"

extern THCState* state;


/*
 * Forward function, project the point features to cells, perform max pooling in every cell 
 * params: 
 *  	  point 	input, all points, Nx2
 *  	  feat_points   input, feature of all points, NxC
 *  	  shape 	input, size of the grid [W, H, D], 3
 *  	  feat_cell     output, feature of all cells, (WxHxD)xC
 *  	  indices     	output, indices of max pooling, saved for back propagation, (WxHxD)xC
 *
 */	
int grid_pooling_cuda_forward( THCudaTensor *point, THCudaTensor *feat_points, THLongTensor *shape, THCudaTensor *feat_cell, THCudaLongTensor *indices)
{

  // data format check
  if (THCudaTensor_nDimension(state, point)!=2 ||  THCudaTensor_nDimension(state, feat_points)!=2){
    printf("Invalid nDimension!\n");
    printf("Expected 2, 2, received %d, %d\n", THCudaTensor_nDimension(state, point), THCudaTensor_nDimension(state, feat_cell));
    return 0;
  }
  // point -> Nx3
  if (THCudaTensor_size(state, point,1)!=3){
    printf("Invalid shape of point!\n");
    return 0;
  }

  grid_pooling_kernel_forward( state, point, feat_points, shape, feat_cell, indices );
  return 1;
}

/*
 * Backward function, back-propagate the loss to the point features
 * params: 
 *  	  grad_output   	input, gradient on the output feature, WxHxC 
 *  	  shape 		input, size of the grid [W, H, D], 3
 *  	  indices     		input, indices of max pooling, WxHxC
 * 	  grad_feat_points 	output, gradient on the features, NxC 
 *
 */	
int grid_pooling_cuda_backward( THCudaTensor *grad_output, THLongTensor *shape, THCudaLongTensor *indices, THCudaTensor *grad_feat_points)
{

  grid_pooling_kernel_backward( state, grad_output, shape, indices, grad_feat_points );

  return 1;
}

