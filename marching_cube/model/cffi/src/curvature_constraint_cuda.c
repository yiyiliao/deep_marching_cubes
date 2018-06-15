#include <THC.h>
#include <THCGeneral.h>
#include <stdio.h>

#include "curvature_constraint_kernel.h"

extern THCState* state;

/*
 * Forward function, calculating the distance from a set of points to one single linesegment 
 * params: 
 * 	  offset 		input, offset map for x,y directions, 2x(W+1)x(H+1)x(D+1) 
 * 	  topolopy 		input, probability for each triangle, (WxHxD)xT'
 * 	  topology_empty 	input, probability for empty topology, (WxHxD) 
 *  	  xTable 	        input, connected __triangles__ in x direction, T'xT' (T'>=T) 
 *  	  yTable 	        input, connected __triangles__ in y direction, T'xT' (T'>=T)
 *  	  zTable 	        input, connected __triangles__ in z direction, T'xT' (T'>=T)
 *  	  innerTable 	        input, connected __triangles__ for one topology within a single cell, T'xT' (T'>=T)
 *  	  loss  		output, smoothness loss on both horizontal and vertical directions, 1 
 *
 */	
int curvature_constraint_cuda_forward(THCudaTensor *offset, THCudaTensor *topology, THCudaTensor *topology_empty, THCudaTensor *xTable, THCudaTensor *yTable, THCudaTensor *zTable, THCudaTensor *innerTable, THCudaTensor *loss )
{
  int W = THCudaTensor_size(state, offset, 1) - 1;
  int H = THCudaTensor_size(state, offset, 2) - 1;
  int D = THCudaTensor_size(state, offset, 3) - 1;
  int T = THCudaTensor_size(state, topology, 1);
  // data format check
  if (THCudaTensor_nDimension(state, offset)!=4 || THCudaTensor_nDimension(state, topology)!=2 || THCudaTensor_nDimension(state, xTable)!=2  || THCudaTensor_nDimension(state, yTable)!=2 || THCudaTensor_nDimension(state, zTable)!=2 || THCudaTensor_nDimension(state, loss)!=1 ){
    printf("Invalid nDimension!\n");
    printf("Expected 4, 2, 2, 2, 2, 1, received %d, %d, %d, %d, %d, %d\n", THCudaTensor_nDimension(state, offset), THCudaTensor_nDimension(state, topology), THCudaTensor_nDimension(state, xTable), THCudaTensor_nDimension(state, yTable), THCudaTensor_nDimension(state, zTable), THCudaTensor_nDimension(state, loss));
    return 0;
  }
  if (THCudaTensor_size(state, offset, 0)!=3 ){
    printf("Invalid shape!\n");
    printf("Expected 3xWxHxD, received %ldx%ldx%ldx%ld\n", THCudaTensor_size(state, offset, 0), THCudaTensor_size(state, offset,1), THCudaTensor_size(state, offset, 2), THCudaTensor_size(state, offset, 3));
    return 0;
  }
  if (THCudaTensor_size(state, topology,0)!=(W*H*D) || THCudaTensor_size(state, topology,1)!=T ){
    printf("Invalid shape!\n");
    printf("Expected %dx%d, received %ldx%ld\n",  W*H*D, T, THCudaTensor_size(state, topology, 0), THCudaTensor_size(state, topology,1));
    return 0;
  }
  if (THCudaTensor_size(state, xTable,0)!=T || THCudaTensor_size(state, xTable,1)!=T ){
    printf("Invalid xTable!\n");
    return 0;
  }
  if (THCudaTensor_size(state, yTable,0)!=T || THCudaTensor_size(state, yTable,1)!=T ){
    printf("Invalid yTable!\n");
    return 0;
  }
  if (THCudaTensor_size(state, zTable,0)!=T || THCudaTensor_size(state, zTable,1)!=T ){
    printf("Invalid yTable!\n");
    return 0;
  }

  curvature_constraint_kernel_forward(state, offset, topology, xTable, yTable, zTable, innerTable, loss );

  
  return 1;

}


/*
 * Backward function, calculating the derivative of the topology with respect to the loss 
 * params: 
 * 	  grad_output 		input, gradient on the output loss, 1
 * 	  offset 		input, offset map for x,y directions, 3x(W+1)x(H+1)x(D+1) 
 * 	  topolopy 		input, probability for each triangle, (WxHxD)xT'
 * 	  topology_empty 	input, probability for empty topology, (WxHxD) 
 *  	  xTable 	        input, connected __triangles__ in x direction, T'xT' (T'>=T) 
 *  	  yTable 	        input, connected __triangles__ in y direction, T'xT' (T'>=T)
 *  	  zTable 	        input, connected __triangles__ in z direction, T'xT' (T'>=T)
 *  	  innerTable 	        input, connected __triangles__ for one topology within a single cell, T'xT' (T'>=T)
 *  	  grad_offset  		output, gradient on the offset, 3x(W+1)x(H+1)x(D+1) 
 *
 */	
int curvature_constraint_cuda_backward(THCudaTensor *grad_output, THCudaTensor *offset, THCudaTensor *topology, THCudaTensor *topology_empty, THCudaTensor *xTable, THCudaTensor *yTable, THCudaTensor *zTable, THCudaTensor *innerTable, THCudaTensor *grad_offset ){
  
  curvature_constraint_kernel_backward(state, grad_output, offset, topology, xTable, yTable, zTable, innerTable, grad_offset);

  return 1;
}
