#include <THC.h>
#include <THCGeneral.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

__constant__ int T = 256;

__constant__ int acceptTopology[48] = {1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 17, 32, 34, 48, 51, 63, 64, 68, 96, 102, 111, 119, 127, 0, // upper 
	                        254, 253, 252, 251, 249, 247, 246, 243, 240, 239, 238, 223, 221, 207, 204, 192, 191, 187, 159, 153, 144, 136, 128, 255}; // bottom



// each row denotes a topology type
// each column denotes one of the vertex of a cell
// 2^8 = 256
__constant__ int occTable[256][8] = {{ 0,  0,  0,  0,  0,  0,  0,  0 },
       { 1,  0,  0,  0,  0,  0,  0,  0 },
       { 0,  1,  0,  0,  0,  0,  0,  0 },
       { 1,  1,  0,  0,  0,  0,  0,  0 },
       { 0,  0,  1,  0,  0,  0,  0,  0 },
       { 1,  0,  1,  0,  0,  0,  0,  0 },
       { 0,  1,  1,  0,  0,  0,  0,  0 },
       { 1,  1,  1,  0,  0,  0,  0,  0 },
       { 0,  0,  0,  1,  0,  0,  0,  0 },
       { 1,  0,  0,  1,  0,  0,  0,  0 },
       { 0,  1,  0,  1,  0,  0,  0,  0 },
       { 1,  1,  0,  1,  0,  0,  0,  0 },
       { 0,  0,  1,  1,  0,  0,  0,  0 },
       { 1,  0,  1,  1,  0,  0,  0,  0 },
       { 0,  1,  1,  1,  0,  0,  0,  0 },
       { 1,  1,  1,  1,  0,  0,  0,  0 },
       { 0,  0,  0,  0,  1,  0,  0,  0 },
       { 1,  0,  0,  0,  1,  0,  0,  0 },
       { 0,  1,  0,  0,  1,  0,  0,  0 },
       { 1,  1,  0,  0,  1,  0,  0,  0 },
       { 0,  0,  1,  0,  1,  0,  0,  0 },
       { 1,  0,  1,  0,  1,  0,  0,  0 },
       { 0,  1,  1,  0,  1,  0,  0,  0 },
       { 1,  1,  1,  0,  1,  0,  0,  0 },
       { 0,  0,  0,  1,  1,  0,  0,  0 },
       { 1,  0,  0,  1,  1,  0,  0,  0 },
       { 0,  1,  0,  1,  1,  0,  0,  0 },
       { 1,  1,  0,  1,  1,  0,  0,  0 },
       { 0,  0,  1,  1,  1,  0,  0,  0 },
       { 1,  0,  1,  1,  1,  0,  0,  0 },
       { 0,  1,  1,  1,  1,  0,  0,  0 },
       { 1,  1,  1,  1,  1,  0,  0,  0 },
       { 0,  0,  0,  0,  0,  1,  0,  0 },
       { 1,  0,  0,  0,  0,  1,  0,  0 },
       { 0,  1,  0,  0,  0,  1,  0,  0 },
       { 1,  1,  0,  0,  0,  1,  0,  0 },
       { 0,  0,  1,  0,  0,  1,  0,  0 },
       { 1,  0,  1,  0,  0,  1,  0,  0 },
       { 0,  1,  1,  0,  0,  1,  0,  0 },
       { 1,  1,  1,  0,  0,  1,  0,  0 },
       { 0,  0,  0,  1,  0,  1,  0,  0 },
       { 1,  0,  0,  1,  0,  1,  0,  0 },
       { 0,  1,  0,  1,  0,  1,  0,  0 },
       { 1,  1,  0,  1,  0,  1,  0,  0 },
       { 0,  0,  1,  1,  0,  1,  0,  0 },
       { 1,  0,  1,  1,  0,  1,  0,  0 },
       { 0,  1,  1,  1,  0,  1,  0,  0 },
       { 1,  1,  1,  1,  0,  1,  0,  0 },
       { 0,  0,  0,  0,  1,  1,  0,  0 },
       { 1,  0,  0,  0,  1,  1,  0,  0 },
       { 0,  1,  0,  0,  1,  1,  0,  0 },
       { 1,  1,  0,  0,  1,  1,  0,  0 },
       { 0,  0,  1,  0,  1,  1,  0,  0 },
       { 1,  0,  1,  0,  1,  1,  0,  0 },
       { 0,  1,  1,  0,  1,  1,  0,  0 },
       { 1,  1,  1,  0,  1,  1,  0,  0 },
       { 0,  0,  0,  1,  1,  1,  0,  0 },
       { 1,  0,  0,  1,  1,  1,  0,  0 },
       { 0,  1,  0,  1,  1,  1,  0,  0 },
       { 1,  1,  0,  1,  1,  1,  0,  0 },
       { 0,  0,  1,  1,  1,  1,  0,  0 },
       { 1,  0,  1,  1,  1,  1,  0,  0 },
       { 0,  1,  1,  1,  1,  1,  0,  0 },
       { 1,  1,  1,  1,  1,  1,  0,  0 },
       { 0,  0,  0,  0,  0,  0,  1,  0 },
       { 1,  0,  0,  0,  0,  0,  1,  0 },
       { 0,  1,  0,  0,  0,  0,  1,  0 },
       { 1,  1,  0,  0,  0,  0,  1,  0 },
       { 0,  0,  1,  0,  0,  0,  1,  0 },
       { 1,  0,  1,  0,  0,  0,  1,  0 },
       { 0,  1,  1,  0,  0,  0,  1,  0 },
       { 1,  1,  1,  0,  0,  0,  1,  0 },
       { 0,  0,  0,  1,  0,  0,  1,  0 },
       { 1,  0,  0,  1,  0,  0,  1,  0 },
       { 0,  1,  0,  1,  0,  0,  1,  0 },
       { 1,  1,  0,  1,  0,  0,  1,  0 },
       { 0,  0,  1,  1,  0,  0,  1,  0 },
       { 1,  0,  1,  1,  0,  0,  1,  0 },
       { 0,  1,  1,  1,  0,  0,  1,  0 },
       { 1,  1,  1,  1,  0,  0,  1,  0 },
       { 0,  0,  0,  0,  1,  0,  1,  0 },
       { 1,  0,  0,  0,  1,  0,  1,  0 },
       { 0,  1,  0,  0,  1,  0,  1,  0 },
       { 1,  1,  0,  0,  1,  0,  1,  0 },
       { 0,  0,  1,  0,  1,  0,  1,  0 },
       { 1,  0,  1,  0,  1,  0,  1,  0 },
       { 0,  1,  1,  0,  1,  0,  1,  0 },
       { 1,  1,  1,  0,  1,  0,  1,  0 },
       { 0,  0,  0,  1,  1,  0,  1,  0 },
       { 1,  0,  0,  1,  1,  0,  1,  0 },
       { 0,  1,  0,  1,  1,  0,  1,  0 },
       { 1,  1,  0,  1,  1,  0,  1,  0 },
       { 0,  0,  1,  1,  1,  0,  1,  0 },
       { 1,  0,  1,  1,  1,  0,  1,  0 },
       { 0,  1,  1,  1,  1,  0,  1,  0 },
       { 1,  1,  1,  1,  1,  0,  1,  0 },
       { 0,  0,  0,  0,  0,  1,  1,  0 },
       { 1,  0,  0,  0,  0,  1,  1,  0 },
       { 0,  1,  0,  0,  0,  1,  1,  0 },
       { 1,  1,  0,  0,  0,  1,  1,  0 },
       { 0,  0,  1,  0,  0,  1,  1,  0 },
       { 1,  0,  1,  0,  0,  1,  1,  0 },
       { 0,  1,  1,  0,  0,  1,  1,  0 },
       { 1,  1,  1,  0,  0,  1,  1,  0 },
       { 0,  0,  0,  1,  0,  1,  1,  0 },
       { 1,  0,  0,  1,  0,  1,  1,  0 },
       { 0,  1,  0,  1,  0,  1,  1,  0 },
       { 1,  1,  0,  1,  0,  1,  1,  0 },
       { 0,  0,  1,  1,  0,  1,  1,  0 },
       { 1,  0,  1,  1,  0,  1,  1,  0 },
       { 0,  1,  1,  1,  0,  1,  1,  0 },
       { 1,  1,  1,  1,  0,  1,  1,  0 },
       { 0,  0,  0,  0,  1,  1,  1,  0 },
       { 1,  0,  0,  0,  1,  1,  1,  0 },
       { 0,  1,  0,  0,  1,  1,  1,  0 },
       { 1,  1,  0,  0,  1,  1,  1,  0 },
       { 0,  0,  1,  0,  1,  1,  1,  0 },
       { 1,  0,  1,  0,  1,  1,  1,  0 },
       { 0,  1,  1,  0,  1,  1,  1,  0 },
       { 1,  1,  1,  0,  1,  1,  1,  0 },
       { 0,  0,  0,  1,  1,  1,  1,  0 },
       { 1,  0,  0,  1,  1,  1,  1,  0 },
       { 0,  1,  0,  1,  1,  1,  1,  0 },
       { 1,  1,  0,  1,  1,  1,  1,  0 },
       { 0,  0,  1,  1,  1,  1,  1,  0 },
       { 1,  0,  1,  1,  1,  1,  1,  0 },
       { 0,  1,  1,  1,  1,  1,  1,  0 },
       { 1,  1,  1,  1,  1,  1,  1,  0 },
       { 0,  0,  0,  0,  0,  0,  0,  1 },
       { 1,  0,  0,  0,  0,  0,  0,  1 },
       { 0,  1,  0,  0,  0,  0,  0,  1 },
       { 1,  1,  0,  0,  0,  0,  0,  1 },
       { 0,  0,  1,  0,  0,  0,  0,  1 },
       { 1,  0,  1,  0,  0,  0,  0,  1 },
       { 0,  1,  1,  0,  0,  0,  0,  1 },
       { 1,  1,  1,  0,  0,  0,  0,  1 },
       { 0,  0,  0,  1,  0,  0,  0,  1 },
       { 1,  0,  0,  1,  0,  0,  0,  1 },
       { 0,  1,  0,  1,  0,  0,  0,  1 },
       { 1,  1,  0,  1,  0,  0,  0,  1 },
       { 0,  0,  1,  1,  0,  0,  0,  1 },
       { 1,  0,  1,  1,  0,  0,  0,  1 },
       { 0,  1,  1,  1,  0,  0,  0,  1 },
       { 1,  1,  1,  1,  0,  0,  0,  1 },
       { 0,  0,  0,  0,  1,  0,  0,  1 },
       { 1,  0,  0,  0,  1,  0,  0,  1 },
       { 0,  1,  0,  0,  1,  0,  0,  1 },
       { 1,  1,  0,  0,  1,  0,  0,  1 },
       { 0,  0,  1,  0,  1,  0,  0,  1 },
       { 1,  0,  1,  0,  1,  0,  0,  1 },
       { 0,  1,  1,  0,  1,  0,  0,  1 },
       { 1,  1,  1,  0,  1,  0,  0,  1 },
       { 0,  0,  0,  1,  1,  0,  0,  1 },
       { 1,  0,  0,  1,  1,  0,  0,  1 },
       { 0,  1,  0,  1,  1,  0,  0,  1 },
       { 1,  1,  0,  1,  1,  0,  0,  1 },
       { 0,  0,  1,  1,  1,  0,  0,  1 },
       { 1,  0,  1,  1,  1,  0,  0,  1 },
       { 0,  1,  1,  1,  1,  0,  0,  1 },
       { 1,  1,  1,  1,  1,  0,  0,  1 },
       { 0,  0,  0,  0,  0,  1,  0,  1 },
       { 1,  0,  0,  0,  0,  1,  0,  1 },
       { 0,  1,  0,  0,  0,  1,  0,  1 },
       { 1,  1,  0,  0,  0,  1,  0,  1 },
       { 0,  0,  1,  0,  0,  1,  0,  1 },
       { 1,  0,  1,  0,  0,  1,  0,  1 },
       { 0,  1,  1,  0,  0,  1,  0,  1 },
       { 1,  1,  1,  0,  0,  1,  0,  1 },
       { 0,  0,  0,  1,  0,  1,  0,  1 },
       { 1,  0,  0,  1,  0,  1,  0,  1 },
       { 0,  1,  0,  1,  0,  1,  0,  1 },
       { 1,  1,  0,  1,  0,  1,  0,  1 },
       { 0,  0,  1,  1,  0,  1,  0,  1 },
       { 1,  0,  1,  1,  0,  1,  0,  1 },
       { 0,  1,  1,  1,  0,  1,  0,  1 },
       { 1,  1,  1,  1,  0,  1,  0,  1 },
       { 0,  0,  0,  0,  1,  1,  0,  1 },
       { 1,  0,  0,  0,  1,  1,  0,  1 },
       { 0,  1,  0,  0,  1,  1,  0,  1 },
       { 1,  1,  0,  0,  1,  1,  0,  1 },
       { 0,  0,  1,  0,  1,  1,  0,  1 },
       { 1,  0,  1,  0,  1,  1,  0,  1 },
       { 0,  1,  1,  0,  1,  1,  0,  1 },
       { 1,  1,  1,  0,  1,  1,  0,  1 },
       { 0,  0,  0,  1,  1,  1,  0,  1 },
       { 1,  0,  0,  1,  1,  1,  0,  1 },
       { 0,  1,  0,  1,  1,  1,  0,  1 },
       { 1,  1,  0,  1,  1,  1,  0,  1 },
       { 0,  0,  1,  1,  1,  1,  0,  1 },
       { 1,  0,  1,  1,  1,  1,  0,  1 },
       { 0,  1,  1,  1,  1,  1,  0,  1 },
       { 1,  1,  1,  1,  1,  1,  0,  1 },
       { 0,  0,  0,  0,  0,  0,  1,  1 },
       { 1,  0,  0,  0,  0,  0,  1,  1 },
       { 0,  1,  0,  0,  0,  0,  1,  1 },
       { 1,  1,  0,  0,  0,  0,  1,  1 },
       { 0,  0,  1,  0,  0,  0,  1,  1 },
       { 1,  0,  1,  0,  0,  0,  1,  1 },
       { 0,  1,  1,  0,  0,  0,  1,  1 },
       { 1,  1,  1,  0,  0,  0,  1,  1 },
       { 0,  0,  0,  1,  0,  0,  1,  1 },
       { 1,  0,  0,  1,  0,  0,  1,  1 },
       { 0,  1,  0,  1,  0,  0,  1,  1 },
       { 1,  1,  0,  1,  0,  0,  1,  1 },
       { 0,  0,  1,  1,  0,  0,  1,  1 },
       { 1,  0,  1,  1,  0,  0,  1,  1 },
       { 0,  1,  1,  1,  0,  0,  1,  1 },
       { 1,  1,  1,  1,  0,  0,  1,  1 },
       { 0,  0,  0,  0,  1,  0,  1,  1 },
       { 1,  0,  0,  0,  1,  0,  1,  1 },
       { 0,  1,  0,  0,  1,  0,  1,  1 },
       { 1,  1,  0,  0,  1,  0,  1,  1 },
       { 0,  0,  1,  0,  1,  0,  1,  1 },
       { 1,  0,  1,  0,  1,  0,  1,  1 },
       { 0,  1,  1,  0,  1,  0,  1,  1 },
       { 1,  1,  1,  0,  1,  0,  1,  1 },
       { 0,  0,  0,  1,  1,  0,  1,  1 },
       { 1,  0,  0,  1,  1,  0,  1,  1 },
       { 0,  1,  0,  1,  1,  0,  1,  1 },
       { 1,  1,  0,  1,  1,  0,  1,  1 },
       { 0,  0,  1,  1,  1,  0,  1,  1 },
       { 1,  0,  1,  1,  1,  0,  1,  1 },
       { 0,  1,  1,  1,  1,  0,  1,  1 },
       { 1,  1,  1,  1,  1,  0,  1,  1 },
       { 0,  0,  0,  0,  0,  1,  1,  1 },
       { 1,  0,  0,  0,  0,  1,  1,  1 },
       { 0,  1,  0,  0,  0,  1,  1,  1 },
       { 1,  1,  0,  0,  0,  1,  1,  1 },
       { 0,  0,  1,  0,  0,  1,  1,  1 },
       { 1,  0,  1,  0,  0,  1,  1,  1 },
       { 0,  1,  1,  0,  0,  1,  1,  1 },
       { 1,  1,  1,  0,  0,  1,  1,  1 },
       { 0,  0,  0,  1,  0,  1,  1,  1 },
       { 1,  0,  0,  1,  0,  1,  1,  1 },
       { 0,  1,  0,  1,  0,  1,  1,  1 },
       { 1,  1,  0,  1,  0,  1,  1,  1 },
       { 0,  0,  1,  1,  0,  1,  1,  1 },
       { 1,  0,  1,  1,  0,  1,  1,  1 },
       { 0,  1,  1,  1,  0,  1,  1,  1 },
       { 1,  1,  1,  1,  0,  1,  1,  1 },
       { 0,  0,  0,  0,  1,  1,  1,  1 },
       { 1,  0,  0,  0,  1,  1,  1,  1 },
       { 0,  1,  0,  0,  1,  1,  1,  1 },
       { 1,  1,  0,  0,  1,  1,  1,  1 },
       { 0,  0,  1,  0,  1,  1,  1,  1 },
       { 1,  0,  1,  0,  1,  1,  1,  1 },
       { 0,  1,  1,  0,  1,  1,  1,  1 },
       { 1,  1,  1,  0,  1,  1,  1,  1 },
       { 0,  0,  0,  1,  1,  1,  1,  1 },
       { 1,  0,  0,  1,  1,  1,  1,  1 },
       { 0,  1,  0,  1,  1,  1,  1,  1 },
       { 1,  1,  0,  1,  1,  1,  1,  1 },
       { 0,  0,  1,  1,  1,  1,  1,  1 },
       { 1,  0,  1,  1,  1,  1,  1,  1 },
       { 0,  1,  1,  1,  1,  1,  1,  1 },
       { 1,  1,  1,  1,  1,  1,  1,  1 }};


__constant__ int vertexTable[8][3]={ {0, 1, 0},
			       {1, 1, 0},
			       {1, 0, 0},
                               {0, 0, 0},
			       {0, 1, 1},
			       {1, 1, 1},
			       {1, 0, 1},
                               {0, 0, 1} };


/**
 * convert the topology probabilites from the occupancy
 * parallel over every cell and every topology
 */
__global__ void occupancy_to_topology_kernel(const float *occupancy, float *topology){
  // int W = gridDim.x;
  int H = gridDim.y;
  int D = gridDim.z;

  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;

  int t = threadIdx.x;
  // return probabilities of all 256 topologies
  int topology_ind = t; 

  float p_occ[2][8];
  for (int v=0; v<8; v++){
    p_occ[0][v] = occupancy[ (i+vertexTable[v][0])*(H+1)*(D+1) + (j+vertexTable[v][1])*(D+1) + k+vertexTable[v][2] ]; 
    p_occ[1][v] = 1-p_occ[0][v]; 
  }


  float p_accumu = 1.0;
  for (int v=0; v<8; v++){
      p_accumu = p_accumu*p_occ[occTable[topology_ind][v]][v]; 
  }
  topology[ (i*H*D+j*D+k)*T + t ] = p_accumu;
}


/**
 * propagate the gradient from the topology probabilities to occupancy status
 * parallel over every cell and every topology
 */
__global__ void grad_occupancy_to_topology_kernel(const float *grad_output, const float *occupancy, float *topology, float *grad_occupancy){
  // int W = gridDim.x;
  int H = gridDim.y;
  int D = gridDim.z;

  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;

  int t = threadIdx.x;
  // return probabilities of all 256 topologies
  int topology_ind = t; 

  float p_occ[2][8];
  for (int v=0; v<8; v++){
    p_occ[0][v] = occupancy[ (i+vertexTable[v][0])*(H+1)*(D+1) + (j+vertexTable[v][1])*(D+1) + k+vertexTable[v][2] ]; 
    p_occ[1][v] = 1-p_occ[0][v]; 
  }


  //float p_accumu = topology[ (i*H+j)*T + t ];
  float grad_accumu = grad_output[ (i*H*D+j*D+k)*T + t ];
  // propagate the gradient to four occupancy corners
  float sign;
  for (int v=0; v<8; v++){
    if (occTable[topology_ind][v]==0){
            sign=1.0;
    }else{
            sign=-1.0;
    } 
  
    // re-calculate the probability excluding the current vertex
    // didn't use p_accumu/p_occ[occTable[t][v]][v] for numerial stability
    // TODO: find a better solution
    float p_accumu = 1.0;
    for (int v_=0; v_<8; v_++){
	if (v_==v) continue;
        p_accumu = p_accumu*p_occ[occTable[topology_ind][v_]][v_]; 
    }
    atomicAdd(&grad_occupancy[ (i+vertexTable[v][0])*(H+1)*(D+1) + (j+vertexTable[v][1])*(D+1) + k+vertexTable[v][2] ], sign*grad_accumu*p_accumu );
  }

}

/*
 * Forward function, compute the topology probability given the occupancy probability 
 * params: 
 * 	  state 	input, THCState
 *  	  occupancy 	input, (W+1)x(H+1)
 *  	  topology     	output, probability of all topologies types we care about (WxH)xT
 *
 */	
void occupancy_to_topology_kernel_forward( THCState *state, THCudaTensor *occupancy, THCudaTensor *topology ){

  int W = THCudaTensor_size(state, occupancy, 0)-1;
  int H = THCudaTensor_size(state, occupancy, 1)-1;
  int D = THCudaTensor_size(state, occupancy, 2)-1;

  int T = THCudaTensor_size(state, topology, 1);

  dim3 dimGrid(W, H, D);
  dim3 dimBlock(T, 1, 1);

  // lauch the kernel
  occupancy_to_topology_kernel<<< dimGrid, dimBlock, 0, THCState_getCurrentStream(state) >>>(
		  THCudaTensor_data(state, occupancy),
		  THCudaTensor_data(state, topology) );

}



/*
 * Backward function, backpropagate the gradient from topology to occupancy 
 * params: 
 * 	  state 		input, THCState
 * 	  grad_output   	input, gradient on the topology probability, (WxH)xT
 *  	  occupancy 		input, (W+1)x(H+1)
 *  	  topology     		input, probability of all topologies types we care about (WxH)xT
 *  	  grad_occupancy   	output, gradient on the occupancy map, (W+1)x(H+1) 
 *
 */	
void occupancy_to_topology_kernel_backward( THCState *state, THCudaTensor *grad_output, THCudaTensor *occupancy, THCudaTensor *topology, THCudaTensor *grad_occupancy ){

  int W = THCudaTensor_size(state, occupancy, 0)-1;
  int H = THCudaTensor_size(state, occupancy, 1)-1;
  int D = THCudaTensor_size(state, occupancy, 2)-1;

  int T = THCudaTensor_size(state, topology, 1);

  dim3 dimGrid(W, H, D);
  dim3 dimBlock(T, 1, 1);

  // lauch the kernel
  grad_occupancy_to_topology_kernel<<< dimGrid, dimBlock, 0, THCState_getCurrentStream(state) >>>(
		  THCudaTensor_data(state, grad_output),
		  THCudaTensor_data(state, occupancy),
		  THCudaTensor_data(state, topology), 
		  THCudaTensor_data(state, grad_occupancy) );
}

#ifdef __cplusplus
}
#endif
