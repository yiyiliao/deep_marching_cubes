#include <THC.h>
#include <THCGeneral.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

__constant__ float grid_size=1.0;


/**
 * perform max-pooling within the cells
 * parallel over each cell and each feature dimension
 */
__global__ void grid_pooling_kernel( const float *point, const float *feat_points, float *feat_cell, long int *indices, const int n  ){

  // cell indices
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;
  // cell size
  // int W = gridDim.x;
  int H = gridDim.y;
  int D = gridDim.z;
  int ind = i*H*D + j*D + k;

  int c = threadIdx.x;
  int C = blockDim.x;
  
  for (int p=0; p<n; p++){
     float px = point[p*3+0];
     float py = point[p*3+1];
     float pz = point[p*3+2];
     // if point is inside of the grid
     if (px >= i && px < i+grid_size && py >= j && py < j+grid_size && pz >= k && pz < k+grid_size){
	  // max-pooling, update feat_cell if the feature is larger than the current feat_cell
	  // can be async for max operation
	  if ( feat_points[p*C + c] > feat_cell[ind*C + c] ){
	     feat_cell[ind*C + c] = feat_points[p*C + c];
	     indices[ind*C + c] = p;
	  }
     }
  }
}

/**
 * back-propagate the loss from the max-pooled feature to point features
 * parallel over each cell and each feature dimension
 */
__global__ void grad_grid_pooling_kernel( const float *grad_output, const long int *indices, float *grad_feat_points  ){

  // cell indices
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;
  // cell size
  // int W = gridDim.x;
  int H = gridDim.y;
  int D = gridDim.z;
  int ind = i*H*D + j*D + k;

  int c = threadIdx.x;
  int C = blockDim.x;

  long int p = indices[ind*C + c];
  if (p < 0) return;

  grad_feat_points[p*C + c] = grad_output[ind*C + c];

}


/*
 * Forward function, project the point features to cells, perform max pooling in every cell 
 * params: 
 *  	  state 	input, THCState
 *  	  point 	input, all points, Nx3
 *  	  feat_points   input, feature of all points, NxC
 *  	  shape 	input, size of the grid [W, H, D], 3
 *  	  feat_cell     output, feature of all cells, (WxHxD)xC
 *  	  indices     	output, indices of max pooling, saved for back propagation, (WxHxD)xC
 *
 */	
void grid_pooling_kernel_forward( THCState *state, THCudaTensor *point, THCudaTensor *feat_points, THLongTensor *shape, THCudaTensor *feat_cell, THCudaLongTensor *indices)
{
  int W = THLongTensor_get1d(shape, 0);
  int H = THLongTensor_get1d(shape, 1);
  int D = THLongTensor_get1d(shape, 2);
  int C = THCudaTensor_size(state,feat_cell,1); 

  dim3 dimGrid(W, H, D);
  dim3 dimBlock(C, 1, 1);

  int n = THCudaTensor_size(state, point, 0);

  grid_pooling_kernel<<< dimGrid, dimBlock, 0, THCState_getCurrentStream(state) >>>(
		  THCudaTensor_data(state, point),
		  THCudaTensor_data(state, feat_points),
		  THCudaTensor_data(state, feat_cell),
		  THCudaLongTensor_data(state, indices), 
		  n);
        
}

/*
 * Backward function, back-propagate the loss to the point features
 * params: 
 *  	  state 	input, THCState
 *  	  grad_output   	input, gradient on the output feature, WxHxC 
 *  	  shape 		input, size of the grid [W, H, D], 3
 *  	  indices     		input, indices of max pooling, WxHxC
 * 	  grad_feat_points 	output, gradient on the features, NxC 
 *
 */	
void grid_pooling_kernel_backward( THCState *state, THCudaTensor *grad_output, THLongTensor *shape, THCudaLongTensor *indices, THCudaTensor *grad_feat_points)
{
  int W = THLongTensor_get1d(shape, 0);
  int H = THLongTensor_get1d(shape, 1);
  int D = THLongTensor_get1d(shape, 2);
  int C = THCudaTensor_size(state,grad_output,1); 

  dim3 dimGrid(W, H, D);
  dim3 dimBlock(C, 1, 1);

  // copy the gradient from each cell to all points
  // according to the max indices
  grad_grid_pooling_kernel<<< dimGrid, dimBlock, 0, THCState_getCurrentStream(state) >>>(
		  THCudaTensor_data(state, grad_output),
		  THCudaLongTensor_data(state, indices), 
		  THCudaTensor_data(state, grad_feat_points));

}

#ifdef __cplusplus
}
#endif
