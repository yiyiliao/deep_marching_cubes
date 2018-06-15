#include <TH/TH.h>
#include <stdio.h>
#include <math.h>

/*
 * Forward function, regularize the neighboring occupancy status to be close 
 * params: 
 *  	  occupancy 	input, (W+1)x(H+1)x(D+1)
 *  	  loss     	output, connectivity loss 
 *
 */	
int occupancy_connectivity_forward( THFloatTensor *occupancy, THFloatTensor *loss ){

  int W = THFloatTensor_size(occupancy, 0);
  int H = THFloatTensor_size(occupancy, 1);
  int D = THFloatTensor_size(occupancy, 2);

  // data format check
  if (THFloatTensor_nDimension(occupancy)!=3 ||  THFloatTensor_nDimension(loss)!=1){
    printf("Invalid nDimension!\n");
    printf("Expected 3, 1, received %d, %d\n", THFloatTensor_nDimension(occupancy), THFloatTensor_nDimension(loss));
    return 0;
  }


  float loss_=0.0;
  for(int i=0; i<W; i++){
    for (int j=0; j<H; j++){
      for (int k=0; k<D; k++){
        float p1 = THFloatTensor_get3d(occupancy, i, j, k); 

        if (j<H-1){
            float p2 = THFloatTensor_get3d(occupancy, i, j+1, k); 
	    // l1 loss
            loss_ += fabs(p1-p2);
        }
        if (i<W-1){
            float p3 = THFloatTensor_get3d(occupancy, i+1, j, k); 
	    // l1 loss
            loss_ += fabs(p1-p3);
        }
        if (k<D-1){
            float p4 = THFloatTensor_get3d(occupancy, i, j, k+1); 
	    // l1 loss
            loss_ += fabs(p1-p4);
        }
      }
    }
  }


  THFloatTensor_set1d(loss, 0, loss_);

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
int occupancy_connectivity_backward( THFloatTensor *grad_output, THFloatTensor *occupancy, THFloatTensor *grad_occupancy ){

  int W = THFloatTensor_size(occupancy, 0);
  int H = THFloatTensor_size(occupancy, 1);
  int D = THFloatTensor_size(occupancy, 2);


  float curr_grad;
  for(int i=0; i<W; i++){
    for (int j=0; j<H; j++){
      for (int k=0; k<D; k++){
        float p1 = THFloatTensor_get3d(occupancy, i, j, k); 

	float sign;
        if (j<H-1){
            float p2 = THFloatTensor_get3d(occupancy, i, j+1, k); 
	    if (p1-p2>0){ sign = 1.0; } else { sign = -1.0; }
            curr_grad = THFloatTensor_get3d(grad_occupancy, i, j, k); 
            THFloatTensor_set3d(grad_occupancy, i, j,   k, curr_grad + sign);
            curr_grad = THFloatTensor_get3d(grad_occupancy, i, j+1, k); 
            THFloatTensor_set3d(grad_occupancy, i, j+1, k, curr_grad - sign);
        }
        if (i<W-1){
            float p3 = THFloatTensor_get3d(occupancy, i+1, j, k); 
	    if (p1-p3>0){ sign = 1.0; } else { sign = -1.0; }
            curr_grad = THFloatTensor_get3d(grad_occupancy, i, j, k); 
            THFloatTensor_set3d(grad_occupancy, i  , j, k, curr_grad + sign);
            curr_grad = THFloatTensor_get3d(grad_occupancy, i+1, j, k); 
            THFloatTensor_set3d(grad_occupancy, i+1, j, k, curr_grad - sign);
        }
        if (k<D-1){
            float p4 = THFloatTensor_get3d(occupancy, i, j, k+1); 
	    if (p1-p4>0){ sign = 1.0; } else { sign = -1.0; }
            curr_grad = THFloatTensor_get3d(grad_occupancy, i, j, k); 
            THFloatTensor_set3d(grad_occupancy, i  , j, k, curr_grad + sign);
            curr_grad = THFloatTensor_get3d(grad_occupancy, i, j, k+1); 
            THFloatTensor_set3d(grad_occupancy, i, j, k+1, curr_grad - sign);
	}
      }
    }
  }

  float grad_output_=THFloatTensor_get1d(grad_output, 0);
  THFloatTensor_mul(grad_occupancy, grad_occupancy, grad_output_);

  return 1;

}

