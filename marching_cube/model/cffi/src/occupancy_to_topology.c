#include <TH/TH.h>
#include <stdio.h>

const int T = 256;


// each row denotes a topology type
// each column denotes one of the vertex of a cell
// 2^8 = 256
static int occTable[256][8] = {{ 0,  0,  0,  0,  0,  0,  0,  0 },
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


static int vertexTable[8][3]={ {0, 1, 0},
			       {1, 1, 0},
			       {1, 0, 0},
                               {0, 0, 0},
			       {0, 1, 1},
			       {1, 1, 1},
			       {1, 0, 1},
                               {0, 0, 1} };


/*
 * Forward function, compute the topology probability given the occupancy probability 
 * params: 
 *  	  occupancy 	input, (W+1)x(H+1)x(D+1)
 *  	  topology     	output, probability of all topologies types we care about (WxHxD)xT
 *
 */	
int occupancy_to_topology_forward( THFloatTensor *occupancy, THFloatTensor *topology ){

  int W = THFloatTensor_size(occupancy, 0)-1;
  int H = THFloatTensor_size(occupancy, 1)-1;
  int D = THFloatTensor_size(occupancy, 2)-1;

  // data format check
  if (THFloatTensor_nDimension(occupancy)!=3 ||  THFloatTensor_nDimension(topology)!=2){
    printf("Invalid nDimension!\n");
    printf("Expected 3, 2, received %d, %d\n", THFloatTensor_nDimension(occupancy), THFloatTensor_nDimension(topology));
    return 0;
  }
  if (THFloatTensor_size(topology,0)!=W*H*D || THFloatTensor_size(topology,1)!=T){
    printf("Invalid shape of topology!\n");
    return 0;
  }

  for(int i=0; i<W; i++){
    for (int j=0; j<H; j++){
      for (int k=0; k<D; k++){
	// read both positive probability and negative probability from the occupancy map
        float p_occ[2][8];
        for (int v=0; v<8; v++){
	  p_occ[0][v] = THFloatTensor_get3d(occupancy, i+vertexTable[v][0], j+vertexTable[v][1], k+vertexTable[v][2]); 
	  p_occ[1][v] = 1-p_occ[0][v]; 
	}
	// get the probability of each topology type from the occupancy status of the corners
	for (int t=0; t<T; t++){
	    int topology_ind = t;
            float p_accumu = 1.0;
            for (int v=0; v<8; v++){
                p_accumu = p_accumu*p_occ[occTable[topology_ind][v]][v]; 
	    }
            THFloatTensor_set2d(topology, i*H*D+j*D+k, t, p_accumu);

	}
      }
    }
  }

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
int occupancy_to_topology_backward( THFloatTensor *grad_output, THFloatTensor *occupancy, THFloatTensor *topology, THFloatTensor *grad_occupancy ){

  int W = THFloatTensor_size(occupancy, 0)-1;
  int H = THFloatTensor_size(occupancy, 1)-1;
  int D = THFloatTensor_size(occupancy, 2)-1;
  for(int i=0; i<W; i++){
    for (int j=0; j<H; j++){
      for (int k=0; k<D; k++){
	// read both positive probability and negative probability from the occupancy map
        float p_occ[2][8];
        for (int v=0; v<8; v++){
	  p_occ[0][v] = THFloatTensor_get3d(occupancy, i+vertexTable[v][0], j+vertexTable[v][1], k+vertexTable[v][2]); 
	  p_occ[1][v] = 1-p_occ[0][v]; 
	}

	for (int t=0; t<T; t++){

	    int topology_ind =  t;
	
            //float p_accumu = THFloatTensor_get2d(topology, i*H+j, t);
            float grad_accumu = THFloatTensor_get2d(grad_output, i*H*D+j*D+k, t);
	    // propagate the gradient to four occupancy corners
	    for (int v=0; v<8; v++){
	    
	      float curr_grad, sign;
	      curr_grad = THFloatTensor_get3d(grad_occupancy, i+vertexTable[v][0], j+vertexTable[v][1], k+vertexTable[v][2]);
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
	      THFloatTensor_set3d(grad_occupancy, i+vertexTable[v][0], j+vertexTable[v][1], k+vertexTable[v][2], curr_grad + sign*grad_accumu*p_accumu );
	    }
	}
      }
    }
  }
  return 1;
}
