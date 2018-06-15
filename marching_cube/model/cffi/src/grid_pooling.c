#include <TH/TH.h>
#include <stdio.h>
#include "commons.h"

const float eps=1e-8;

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
int grid_pooling_forward( THFloatTensor *point, THFloatTensor *feat_points, THLongTensor *shape, THFloatTensor *feat_cell, THLongTensor *indices)
{
  long int W = THLongTensor_get1d(shape, 0);
  long int H = THLongTensor_get1d(shape, 1);
  long int D = THLongTensor_get1d(shape, 2);
  int C = THFloatTensor_size(feat_cell,1); 

  // data format check
  if (THFloatTensor_nDimension(point)!=2 ||  THFloatTensor_nDimension(feat_points)!=2){
    printf("Invalid nDimension!\n");
    printf("Expected 2, 2, received %d, %d\n", THFloatTensor_nDimension(point), THFloatTensor_nDimension(feat_cell));
    return 0;
  }
  // point -> Nx3
  if (THFloatTensor_size(point,1)!=3){
    printf("Invalid shape of point!\n");
    return 0;
  }

  for (int i=0; i<W; i++){
    for (int j=0; j<H; j++){
      for (int k=0; k<D; k++){
        THLongTensor *point_indices = points_in_grid(point, i, j, k);
	// if the grid is empty, directly assign constant values  
	if (THLongTensor_nDimension(point_indices)>0) {
	  // if the grid is not empty 
          THFloatTensor *feat_grid = THFloatTensor_new();
	  THFloatTensor_indexSelect(feat_grid, feat_points, 0, point_indices);

	  // max pooling over all points
	  THLongTensor *indices_max = THLongTensor_new();
          THFloatTensor *feat_max = THFloatTensor_new();
	  THFloatTensor_max(feat_max, indices_max, feat_grid, 0, 1);


	  // change local max indices to global max indices
	  for (int l=0; l<C; l++){
	     int local_ind = THLongTensor_get2d(indices_max, 0, l);
	     int global_ind = THLongTensor_get1d(point_indices, local_ind);
	     THLongTensor_set2d(indices_max, 0, l, global_ind);
	  }

	  // copy to the specific cell 
	  THLongTensor *ind = THLongTensor_newWithSize1d(1);
	  THLongTensor_set1d(ind, 0, i*H*D + j*D + k);
	  THFloatTensor_indexCopy(feat_cell, 0, ind, feat_max);
	  THLongTensor_indexCopy(indices, 0, ind, indices_max);

	  THLongTensor_free(ind);
	  THFloatTensor_free(feat_max);
	  THLongTensor_free(indices_max);
	  THFloatTensor_free(feat_grid);
	}
	THLongTensor_free(point_indices);
      }
    }
  }
        
  return 1;
}

/*
 * Backward function, back-propagate the loss to the point features
 * params: 
 *  	  grad_output   	input, gradient on the output feature, WxHxC 
 *  	  indices     		input, indices of max pooling, WxHxC
 * 	  grad_feat_points 	output, gradient on the features, NxC 
 *
 */	
int grid_pooling_backward( THFloatTensor *grad_output, THLongTensor *indices, THFloatTensor *grad_feat_points)
{
  int N = THFloatTensor_size(grad_output,0); 
  int C = THFloatTensor_size(grad_output,1); 
  // copy the gradient from each cell to all points
  // according to the max indices
  for (int i=0; i<N; i++){
    for (int k=0; k<C; k++){
      int ind = THLongTensor_get2d(indices, i, k);
      if (ind==-1) continue;
      float grad = THFloatTensor_get2d(grad_output, i, k);
      THFloatTensor_set2d(grad_feat_points, ind, k, grad);
    }
  }

  return 1;
}

