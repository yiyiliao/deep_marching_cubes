#include <TH/TH.h>
#include <stdio.h>
#include <math.h>
#include "commons.h"

const float eps=1e-6;

const float thres=1e-4;


static int acceptTopologyWithFlip[2][96]={ {1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 25, 31, 32, 34, 35, 38, 47, 48, 49, 50, 51, 55, 59, 63, 64, 68, 70, 76, 79, 96, 98, 100, 102, 103, 110, 111, 112, 115, 118, 119, 127, 0, 255, 128, 136, 137, 140, 143, 144, 145, 152, 153, 155, 157, 159, 176, 179, 185, 187, 191, 192, 196, 200, 204, 205, 206, 207, 208, 217, 220, 221, 223, 224, 230, 236, 238, 239, 240, 241, 242, 243, 244, 246, 247, 248, 249, 251, 252, 253, 254},
				    {1, 1, 2, 1, 2, 3, 1, 2, 3, 2, 3, 3, 2, 1, 2, 3, 3, 3, 1, 2, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 1, 2, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 1, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 2, 3, 3, 2, 3, 2, 1, 3, 2, 1, 2, 1, 1}};



/**
 * check the intersection between two integer lists 
 * param:
 * 	array1 		integer list denoting the vertex indices on a single face, length 4
 * 	array2 		integer list denoting the vertex indices of a triangle, length 3
 * return:
 * 	out		intersected vertex indices, padded with -1 to a fixed length, length 3
 */
int *intersection(int *array1, int *array2){

    int count = 0;

    int *out = malloc(3 * sizeof(int) );
    out[0] = -1;
    out[1] = -1;
    out[2] = -1;
    

    for (int i=0; i<4; i++){
	for (int j=0; j<3; j++){
	    if (array2[j]==array1[i]){
		out[count] = array1[i]; 
		count ++;
	    }
	}
    }
    return out;
}

/**
 * return the vertex indices on a given surface of a cell
 */
int* get_vertices_on_face(int r){
    int vertices_on_location[6][4] = { {5, 9, 1, 10},
                                   {7, 8, 3, 11},
                                   {4, 9, 0, 8},
                                   {6, 10, 2, 11},
                                   {4, 5, 6, 7},
                                   {0, 1, 2, 3} };
    int *row = malloc(4 * sizeof(int));
    for (int i=0; i<4; i++){
      row[i]=vertices_on_location[r][i];
    }
    return row;
}

/** 
 * offset_to_normals, return normal vectors of all triangles (NOT topologies)
 * params:
 * 	normal 		output
 * 	offset 		input
 *	i_ 		input, index of the cell
 *	j_ 		input, index of the cell
 *	k_ 		input, index of the cell
 *	location	input, indicating the relative location of the current cell in the pairwise loss
 * 				0: x1 
 * 				1: x2 
 * 				2: y1 
 * 				3: y2 
 * 				4: z1 
 * 				5: z2 
 * 				6: dummy case for inner cell loss 
 * 				7: dummy case for inner cell loss 
 */
void offset_to_normals(THFloatTensor *normal, THFloatTensor *offset, int i_, int j_, int k_, int location){


  THFloatTensor *vertices = offset_to_vertices(offset, i_, j_, k_);

  int *vertices_on_face = get_vertices_on_face(location); 


  // TODO: change harding coding below
  // TODO: doesn't support duplicate checking
  int tri_cnt = 0;
  for (int i = 0; i < 96; i++){
      int top_ind = acceptTopologyWithFlip[0][i]; 
      int num_triangle = acceptTopologyWithFlip[1][i];
      for (int tri_ind = 0; tri_ind<num_triangle; tri_ind++){
	  // get the indices of the triangle vertices
	  int triangle[3] = {triTable[top_ind][tri_ind*3], triTable[top_ind][tri_ind*3+1], triTable[top_ind][tri_ind*3+2]};
	  
	  // check if the triangle has a line on the face we care about
	  // simply assign a dummy normal vector if not
          int *inter_ind = intersection(vertices_on_face, triangle);


	  // location > 5 means inner loss case instead of x, y, z direction
	  if ( location>5 || ( location <=5 && inter_ind[0]>-1 && inter_ind[1]>-1 && inter_ind[2]==-1) ){

	    // when there is inside/outside information, 
	    // we should directly take the order defined by the look-up-table
	    int a, b, c;
	    a = triangle[0];
	    b = triangle[1];
	    c = triangle[2];
		
	    THFloatTensor *vec1 = THFloatTensor_newWithSize1d(3);
	    THFloatTensor *vec2 = THFloatTensor_newWithSize1d(3);
	    THFloatTensor_set1d(vec1, 0, THFloatTensor_get2d(vertices, 0, b) - THFloatTensor_get2d(vertices, 0, a));
	    THFloatTensor_set1d(vec1, 1, THFloatTensor_get2d(vertices, 1, b) - THFloatTensor_get2d(vertices, 1, a));
	    THFloatTensor_set1d(vec1, 2, THFloatTensor_get2d(vertices, 2, b) - THFloatTensor_get2d(vertices, 2, a));

	    THFloatTensor_set1d(vec2, 0, THFloatTensor_get2d(vertices, 0, c) - THFloatTensor_get2d(vertices, 0, a));
	    THFloatTensor_set1d(vec2, 1, THFloatTensor_get2d(vertices, 1, c) - THFloatTensor_get2d(vertices, 1, a));
	    THFloatTensor_set1d(vec2, 2, THFloatTensor_get2d(vertices, 2, c) - THFloatTensor_get2d(vertices, 2, a));

	    THFloatTensor *cross = THFloatTensor_newWithSize1d(3);
	    THFloatTensor_cross(cross, vec1, vec2, 0);

	    THFloatTensor_set2d(normal, 0, tri_cnt, THFloatTensor_get1d(cross, 0));
	    THFloatTensor_set2d(normal, 1, tri_cnt, THFloatTensor_get1d(cross, 1));
	    THFloatTensor_set2d(normal, 2, tri_cnt, THFloatTensor_get1d(cross, 2));

	    THFloatTensor_free(vec1);
	    THFloatTensor_free(vec2);
	    THFloatTensor_free(cross);
	  }
	  else{
            THFloatTensor_set2d(normal, 0, tri_cnt, 1);
            THFloatTensor_set2d(normal, 1, tri_cnt, 1);
            THFloatTensor_set2d(normal, 2, tri_cnt, 1);
	  }

	  free(inter_ind);
	  
	  tri_cnt ++;
      }
  }
  THFloatTensor_free(vertices);
  free(vertices_on_face);

}



/**
 * calculate d(normalized normal vector)/d(normal vector)
 */
void grad_normalized_to_normal(THFloatTensor *grad_normal, THFloatTensor *normalized_normal, THFloatTensor *length){
  long int T = THFloatTensor_size(grad_normal, 1);
  // 
  THFloatTensor *grad_abs = THFloatTensor_newWithSize2d(3, T);
  THFloatTensor_abs(grad_abs, grad_normal);
  if (THFloatTensor_sumall(grad_abs)<1e-3){
	  return;
  }
  THFloatTensor *eye = THFloatTensor_newWithSize2d(3, 3);
  THFloatTensor_eye(eye, 3, 3);

  for (int i=0; i<THFloatTensor_size(grad_normal, 1); i++){
    float l = THFloatTensor_get2d(length, 0, i);

    THFloatTensor *n = THFloatTensor_newSelect(normalized_normal, 1, i);
    THFloatTensor *nt = THFloatTensor_newSelect(normalized_normal, 1, i);

    THFloatTensor_unsqueeze1d(n, n, 1);
    THFloatTensor_unsqueeze1d(nt, nt, 0);
    
    THFloatTensor *prod = THFloatTensor_newWithSize2d(3, 3);
    THFloatTensor_zero(prod);
    THFloatTensor_addmm(prod, 0.0, prod, -1.0, n, nt);

    THFloatTensor_cadd(prod, prod, l*l, eye);
    THFloatTensor_div(prod, prod, l*l*l);

    THFloatTensor *gradn = THFloatTensor_newSelect(grad_normal, 1, i);
    // IMPORTANT: Bug with resize2d? value changed after that
    //THFloatTensor_resize2d(gradn, 1, 2);
    THFloatTensor_unsqueeze1d(gradn, gradn, 0);
    THFloatTensor_addmm(gradn, 0.0, gradn, 1.0, gradn, prod);

    THFloatTensor_free(gradn);
    THFloatTensor_free(prod);
    THFloatTensor_free(n);
    THFloatTensor_free(nt);
  }
  THFloatTensor_free(eye);
  THFloatTensor_free(grad_abs);
}


/**
 * calculate dn/dpb
 *  	0  		c3-a3  		-(c2-a2)
 *  	-(c3-a3)  	0 		c1-a1
 *  	c2-a2 		-(c1-a1)	0
 */
THFloatTensor *dn_dpb( THFloatTensor *vertices, int a, int c ){
  THFloatTensor *w = THFloatTensor_newWithSize2d(3, 3);
  THFloatTensor_zero(w);

  float d3 = THFloatTensor_get2d(vertices, 2, c) - THFloatTensor_get2d(vertices, 2, a);
  float d2 = THFloatTensor_get2d(vertices, 1, c) - THFloatTensor_get2d(vertices, 1, a);
  float d1 = THFloatTensor_get2d(vertices, 0, c) - THFloatTensor_get2d(vertices, 0, a);
  THFloatTensor_set2d(w, 0, 1, d3);
  THFloatTensor_set2d(w, 0, 2, -d2);
  THFloatTensor_set2d(w, 1, 0, -d3);
  THFloatTensor_set2d(w, 1, 2, d1);
  THFloatTensor_set2d(w, 2, 0, d2);
  THFloatTensor_set2d(w, 2, 1, -d1);

  return w;
}

/**
 * calculate dn/dpc
 * 	0  		-(b3-a3)  	b2-a2
 * 	(b3-a3)  	0 		-(b1-a1)
 * 	-(b2-a2) 	b1-a1		0
 */
THFloatTensor *dn_dpc( THFloatTensor *vertices, int a, int b ){
  THFloatTensor *w = THFloatTensor_newWithSize2d(3, 3);
  THFloatTensor_zero(w);

  float d3 = THFloatTensor_get2d(vertices, 2, b) - THFloatTensor_get2d(vertices, 2, a);
  float d2 = THFloatTensor_get2d(vertices, 1, b) - THFloatTensor_get2d(vertices, 1, a);
  float d1 = THFloatTensor_get2d(vertices, 0, b) - THFloatTensor_get2d(vertices, 0, a);
  THFloatTensor_set2d(w, 0, 1, -d3);
  THFloatTensor_set2d(w, 0, 2, d2);
  THFloatTensor_set2d(w, 1, 0, d3);
  THFloatTensor_set2d(w, 1, 2, -d1);
  THFloatTensor_set2d(w, 2, 0, -d2);
  THFloatTensor_set2d(w, 2, 1, d1);

  return w;
}

/**
 * calculate dn/dpa
 * 	 0  		b3-c3  		-(b2-c2)
 * 	 -(b3-c3)  	0 		b1-c1	
 * 	 b2-c2 		-(b1-c1)	0
 */
THFloatTensor *dn_dpa( THFloatTensor *vertices, int b, int c ){
  THFloatTensor *w = THFloatTensor_newWithSize2d(3, 3);
  THFloatTensor_zero(w);

  float d3 = THFloatTensor_get2d(vertices, 2, b) - THFloatTensor_get2d(vertices, 2, c);
  float d2 = THFloatTensor_get2d(vertices, 1, b) - THFloatTensor_get2d(vertices, 1, c);
  float d1 = THFloatTensor_get2d(vertices, 0, b) - THFloatTensor_get2d(vertices, 0, c);
  THFloatTensor_set2d(w, 0, 1, d3);
  THFloatTensor_set2d(w, 0, 2, -d2);
  THFloatTensor_set2d(w, 1, 0, -d3);
  THFloatTensor_set2d(w, 1, 2, d1);
  THFloatTensor_set2d(w, 2, 0, d2);
  THFloatTensor_set2d(w, 2, 1, -d1);

  return w;
}

/**
 * grad_normal_to_offset
 */
void grad_normal_to_offset(THFloatTensor *grad_offset, THFloatTensor *grad_normal, THFloatTensor *offset, int i_, int j_, int k_, int location){

  THFloatTensor *vertices = offset_to_vertices(offset, i_, j_, k_);

  int *vertices_on_face = get_vertices_on_face(location); 

  int tri_cnt = 0;
  for (int i = 0; i < 96; i++){
      int top_ind = acceptTopologyWithFlip[0][i]; 
      int num_triangle = acceptTopologyWithFlip[1][i];
      for (int tri_ind = 0; tri_ind<num_triangle; tri_ind++){

	  // get the gradient on the normal vector of the current triangle
	  THFloatTensor *grad_tri = THFloatTensor_newSelect(grad_normal, 1, tri_cnt);
    	  THFloatTensor_unsqueeze1d(grad_tri, grad_tri, 0);

	  // get the indices of the triangle vertices
	  int triangle[3] = {triTable[top_ind][tri_ind*3], triTable[top_ind][tri_ind*3+1], triTable[top_ind][tri_ind*3+2]};
	  
	  // check if the triangle has a line on the face we care about
	  // simply assign a dummy normal vector if not
          int *inter_ind = intersection(vertices_on_face, triangle);


	  // location > 5 means inner loss case instead of x, y, z direction
	  if ( location>5 || ( location <=5 && inter_ind[0]>-1 && inter_ind[1]>-1 && inter_ind[2]==-1) ){

	    // when there is inside/outside information, 
	    // we should directly take the order defined by the look-up-table
	    int a, b, c;
	    a = triangle[0];
	    b = triangle[1];
	    c = triangle[2];

	    float curr_grad;
	    // dn_da
  	    THFloatTensor *dn_da = dn_dpa(vertices, b, c);
	    THFloatTensor *da = THFloatTensor_newWithSize2d(1, 3);
	    THFloatTensor_zero(da);
	    THFloatTensor_addmm(da, 0.0, da, 1.0, grad_tri, dn_da);


            curr_grad = THFloatTensor_get4d( grad_offset, vertices_to_offset[a][0], 
  	    	      				     vertices_to_offset[a][1] + i_,
  	    	                   		     vertices_to_offset[a][2] + j_,
  	    					     vertices_to_offset[a][3] + k_);
            THFloatTensor_set4d( grad_offset, vertices_to_offset[a][0], 
  	    	      		   vertices_to_offset[a][1] + i_,
  	    	                   vertices_to_offset[a][2] + j_, 
  	    			   vertices_to_offset[a][3] + k_,
  	    			   curr_grad - THFloatTensor_get2d(da, 0, vertices_to_offset[a][0]) );
	    // dn_db
  	    THFloatTensor *dn_db = dn_dpb(vertices, a, c);
	    THFloatTensor *db = THFloatTensor_newWithSize2d(1, 3);
	    THFloatTensor_zero(db);
	    THFloatTensor_addmm(db, 0.0, db, 1.0, grad_tri, dn_db);


            curr_grad = THFloatTensor_get4d( grad_offset, vertices_to_offset[b][0], 
  	    	      				     vertices_to_offset[b][1] + i_,
  	    	                   		     vertices_to_offset[b][2] + j_,
  	    					     vertices_to_offset[b][3] + k_);
            THFloatTensor_set4d( grad_offset, vertices_to_offset[b][0], 
  	    	      		   vertices_to_offset[b][1] + i_,
  	    	                   vertices_to_offset[b][2] + j_, 
  	    			   vertices_to_offset[b][3] + k_,
  	    			   curr_grad - THFloatTensor_get2d(db, 0, vertices_to_offset[b][0]) );
	    // dn_dc
  	    THFloatTensor *dn_dc = dn_dpa(vertices, a, b);
	    THFloatTensor *dc = THFloatTensor_newWithSize2d(1, 3);
	    THFloatTensor_zero(dc);
	    THFloatTensor_addmm(dc, 0.0, dc, 1.0, grad_tri, dn_dc);


            curr_grad = THFloatTensor_get4d( grad_offset, vertices_to_offset[c][0], 
  	    	      				     vertices_to_offset[c][1] + i_,
  	    	                   		     vertices_to_offset[c][2] + j_,
  	    					     vertices_to_offset[c][3] + k_);
            THFloatTensor_set4d( grad_offset, vertices_to_offset[c][0], 
  	    	      		   vertices_to_offset[c][1] + i_,
  	    	                   vertices_to_offset[c][2] + j_, 
  	    			   vertices_to_offset[c][3] + k_,
  	    			   curr_grad - THFloatTensor_get2d(dc, 0, vertices_to_offset[c][0]) );

	    THFloatTensor_free(dn_dc);
	    THFloatTensor_free(dn_db);
	    THFloatTensor_free(dn_da);
	    THFloatTensor_free(dc);
	    THFloatTensor_free(db);
	    THFloatTensor_free(da);
	  }

	  THFloatTensor_free(grad_tri);

	  free(inter_ind);
	  
	  tri_cnt ++;
      }
  }
  THFloatTensor_free(vertices);
  free(vertices_on_face);

}

/**
 * calculate the loss between two neighboring cells
 * params:
 * 	offset 		the vertex displacement field of the full grid
 * 	topolopy 	input, probability for each triangle'
 * 	topology_empty 	input, probability for empty topology 
 * 	mask 		mask denoting if two topogolies have connected triangles or not	
 * 	i1 		the index of a single cell
 * 	j1 		the index of a single cell
 * 	k1 		the index of a single cell
 * 	direction	a integer denoting the neighoring relationship between two cells 
 * 				0: two cells adajecent in x direction
 * 				1: two cells adajecent in y direction
 * 				2: two cells adajecent in z direction
 * 				3: dummy label for inner cell loss
 */
float pairwise_loss(THFloatTensor *offset, THFloatTensor *topology, THFloatTensor *topology_empty, THFloatTensor *mask, int i1, int j1, int k1, int direction){
  long int H = THFloatTensor_size(offset, 2) - 1;
  long int D = THFloatTensor_size(offset, 3) - 1;
  long int T = THFloatTensor_size(topology, 1);
  
  int i2=0, j2=0, k2=0, ind1=0, ind2=0;
  // x direction
  if (direction==0){
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1+H*D;
       i2 = i1+1;
       j2 = j1;
       k2 = k1;
  }
  // y direction
  else if (direction==1){
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1+H;
       i2 = i1;
       j2 = j1+1;
       k2 = k1;
  }
  // z direction
  else if (direction==2){
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1+1;
       i2 = i1;
       j2 = j1;
       k2 = k1+1;
  }
  // inner cell 
  else if (direction==3){
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1;
       i2 = i1;
       j2 = j1;
       k2 = k1;
  }

  // for efficiency consideration, skip if both cells are predicted as empty
  // make sure that the last class is empty
  if (THFloatTensor_get1d(topology_empty, ind1) * THFloatTensor_get1d(topology_empty, ind2)> 0.9) {
    return 0.0;
  }

  THFloatTensor *zero_ = THFloatTensor_newWithSize2d(T, T);
  THFloatTensor_zero(zero_);

  THFloatTensor *p1 = THFloatTensor_newSelect(topology, 0, ind1);
  THFloatTensor *p2 = THFloatTensor_newSelect(topology, 0, ind2);

  THFloatTensor_resize2d(p1, T, 1);
  THFloatTensor_resize2d(p2, 1, T);

  // outer product
  THFloatTensor *weight = THFloatTensor_newWithSize2d(T, T);
  THFloatTensor_zero(weight);
  THFloatTensor_addmm(weight, 1.0, zero_, 1.0, p1, p2);

  // multiplied by the binary connection mask
  THFloatTensor_cmul(weight, weight, mask);

  // get normal vector in both grids
  THFloatTensor *norm1 = THFloatTensor_newWithSize2d( 3, T );
  THFloatTensor *norm2 = THFloatTensor_newWithSize2d( 3, T );
  offset_to_normals(norm1, offset, i1, j1, k1, direction*2);
  offset_to_normals(norm2, offset, i2, j2, k2, direction*2+1);

  // normalize to unit vectors
  THFloatTensor *norm1_l2 = THFloatTensor_newWithSize2d(1, T);
  THFloatTensor *norm2_l2 = THFloatTensor_newWithSize2d(1, T);
  THFloatTensor_norm(norm1_l2, norm1, 2, 0, 1);
  THFloatTensor_norm(norm2_l2, norm2, 2, 0, 1);
  // expand to 3xT
  THLongStorage *storage = THFloatTensor_newSizeOf(norm1);
  THFloatTensor *length1 = THFloatTensor_newWithSize2d(3, T);
  THFloatTensor *length2 = THFloatTensor_newWithSize2d(3, T);
  THFloatTensor_expand(length1, norm1_l2, storage);
  THFloatTensor_expand(length2, norm2_l2, storage);
  //THFloatTensor_cat(length1, norm1_l2, norm1_l2, 0);
  //THFloatTensor_cat(length2, norm2_l2, norm2_l2, 0);

  THFloatTensor_cdiv(norm1, norm1, length1);
  THFloatTensor_cdiv(norm2, norm2, length2);

  // calcuate graph regularization loss
  // the following equations are equal as N1 and N2 are normalized, therefore N1*N1.t()=I
  // trace( N1*D1*N1.t() + N2*D2*N2.t() - 2*N1*W*N2.t() )
  // 2*sum(W) - 2*trace(N1*W*N2.t())
  THFloatTensor *tmp3_ =  THFloatTensor_newWithSize2d(3, T);
  THFloatTensor *tmp3 =  THFloatTensor_newWithSize2d(3, 3);
  THFloatTensor_addmm(tmp3_, 0.0, tmp3_, 1.0, norm1, weight);
  THFloatTensor_transpose(norm2, norm2, 0, 1);
  THFloatTensor_addmm(tmp3, 0.0, tmp3, 1.0, tmp3_, norm2);

  float loss = 2*THFloatTensor_sumall(weight) - 2*THFloatTensor_trace(tmp3);

  THFloatTensor_free(tmp3_);
  THFloatTensor_free(tmp3);
  THFloatTensor_free(length1);
  THFloatTensor_free(length2);
  THLongStorage_free(storage);
  THFloatTensor_free(norm1_l2);
  THFloatTensor_free(norm2_l2);
  THFloatTensor_free(norm1);
  THFloatTensor_free(norm2);
  THFloatTensor_free(weight);
  THFloatTensor_free(p1);
  THFloatTensor_free(p2);
  THFloatTensor_free(zero_);
  
  return loss;
}

/**
 * calculate the gradient back-propagated to the offset
 * 	offset 		the vertex displacement field of the full grid
 * 	topology	the topology probability
 * 	topology_empty 	the probability for assigning the empty topology
 * 	grad_offset	gradient on the offset
 * 	mask 		mask denoting if two topogolies have connected triangles or not	
 * 	i1 		the index of a single cell
 * 	j1 		the index of a single cell
 * 	k1 		the index of a single cell
 * 	direction	a integer denoting the neighoring relationship between two cells 
 * 				0: two cells adajecent in x direction
 * 				1: two cells adajecent in y direction
 * 				2: two cells adajecent in z direction
 * 				3: dummy label for inner cell loss
 */
void pairwise_grad(THFloatTensor *offset, THFloatTensor *topology, THFloatTensor *topology_empty, THFloatTensor *grad_offset, THFloatTensor *mask, int i1, int j1, int k1, int direction){
  long int H = THFloatTensor_size(offset, 2) - 1;
  long int D = THFloatTensor_size(offset, 3) - 1;
  long int T = THFloatTensor_size(topology, 1);
  
  int i2=0, j2=0, k2=0, ind1=0, ind2=0;
  // x direction
  if (direction==0){
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1+H*D;
       i2 = i1+1;
       j2 = j1;
       k2 = k1;
  }
  // y direction
  else if (direction==1){
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1+H;
       i2 = i1;
       j2 = j1+1;
       k2 = k1;
  }
  // z direction
  else if (direction==2){
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1+1;
       i2 = i1;
       j2 = j1;
       k2 = k1+1;
  }
  // inner cell 
  else if (direction==3){
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1;
       i2 = i1;
       j2 = j1;
       k2 = k1;
  }
  // for efficiency consideration, skip if both cells are predicted as empty
  // make sure that the last class is empty
  if (THFloatTensor_get1d(topology_empty, ind1) * THFloatTensor_get1d(topology_empty, ind2)> 0.9) {
    return;
  }

  THFloatTensor *p1 = THFloatTensor_newSelect(topology, 0, ind1);
  THFloatTensor *p2 = THFloatTensor_newSelect(topology, 0, ind2);

  THFloatTensor_resize2d(p1, T, 1);
  THFloatTensor_resize2d(p2, 1, T);

  // outer product
  THFloatTensor *weight = THFloatTensor_newWithSize2d(T, T);
  THFloatTensor_zero(weight);
  THFloatTensor_addmm(weight, 0.0, weight, 1.0, p1, p2);

  // multiplied by the binary connection mask
  THFloatTensor_cmul(weight, weight, mask);

  // get normal vector in both grids
  // get the vertices order for back propagation
  THFloatTensor *norm1 = THFloatTensor_newWithSize2d( 3, T );
  THFloatTensor *norm2 = THFloatTensor_newWithSize2d( 3, T );
  offset_to_normals(norm1, offset, i1, j1, k1, direction*2);
  offset_to_normals(norm2, offset, i2, j2, k2, direction*2+1);

  // normalize to unit vectors
  THFloatTensor *norm1_l2 = THFloatTensor_newWithSize2d(1, T);
  THFloatTensor *norm2_l2 = THFloatTensor_newWithSize2d(1, T);
  THFloatTensor_norm(norm1_l2, norm1, 2, 0, 1);
  THFloatTensor_norm(norm2_l2, norm2, 2, 0, 1);

  // expand to 3xT
  THLongStorage *storage = THFloatTensor_newSizeOf(norm1);
  THFloatTensor *length1 = THFloatTensor_newWithSize2d(3, T);
  THFloatTensor *length2 = THFloatTensor_newWithSize2d(3, T);
  THFloatTensor_expand(length1, norm1_l2, storage);
  THFloatTensor_expand(length2, norm2_l2, storage);

  THFloatTensor *normalized_norm1 = THFloatTensor_newWithSize2d(3, T);
  THFloatTensor *normalized_norm2 = THFloatTensor_newWithSize2d(3, T);
  THFloatTensor_cdiv(normalized_norm1, norm1, length1);
  THFloatTensor_cdiv(normalized_norm2, norm2, length2);

  // dL/dn
  THFloatTensor *grad_norm1 = THFloatTensor_newWithSize2d(THFloatTensor_size(norm1, 0), THFloatTensor_size(norm1, 1));
  THFloatTensor *grad_norm2 = THFloatTensor_newWithSize2d(THFloatTensor_size(norm2, 0), THFloatTensor_size(norm2, 1));
  THFloatTensor_zero(grad_norm1);
  THFloatTensor_zero(grad_norm2);

  THFloatTensor_addmm(grad_norm2, 0.0, grad_norm2, -2.0, normalized_norm1, weight);
  THFloatTensor_transpose(weight, weight, 0, 1);
  THFloatTensor_addmm(grad_norm1, 0.0, grad_norm1, -2.0, normalized_norm2, weight);

  // dn/dn'
  grad_normalized_to_normal(grad_norm1, norm1, length1);
  grad_normalized_to_normal(grad_norm2, norm2, length2);

  // dn'/dp
  grad_normal_to_offset(grad_offset, grad_norm1, offset, i1, j1, k1, direction*2);
  grad_normal_to_offset(grad_offset, grad_norm2, offset, i2, j2, k2, direction*2 + 1);

  THFloatTensor_free(grad_norm1);
  THFloatTensor_free(grad_norm2);
  THFloatTensor_free(normalized_norm1);
  THFloatTensor_free(normalized_norm2);
  THFloatTensor_free(length1);
  THFloatTensor_free(length2);
  THLongStorage_free(storage);
  THFloatTensor_free(norm1_l2);
  THFloatTensor_free(norm2_l2);
  THFloatTensor_free(norm1);
  THFloatTensor_free(norm2);
  THFloatTensor_free(weight);
  THFloatTensor_free(p1);
  THFloatTensor_free(p2);
}


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
int curvature_constraint_forward(THFloatTensor *offset, THFloatTensor *topology, THFloatTensor *topology_empty, THFloatTensor *xTable, THFloatTensor *yTable, THFloatTensor *zTable, THFloatTensor *innerTable, THFloatTensor *loss )
{
  int C = THFloatTensor_size(offset, 0);
  int W = THFloatTensor_size(offset, 1) - 1;
  int H = THFloatTensor_size(offset, 2) - 1;
  int D = THFloatTensor_size(offset, 3) - 1;
  int T = THFloatTensor_size(topology, 1);
  // data format check
  if (THFloatTensor_nDimension(offset)!=4 || THFloatTensor_nDimension(topology)!=2 || THFloatTensor_nDimension(xTable)!=2  || THFloatTensor_nDimension(yTable)!=2 || THFloatTensor_nDimension(zTable)!=2 || THFloatTensor_nDimension(loss)!=1 ){
    printf("Invalid nDimension!\n");
    printf("Expected 4, 2, 2, 2, 2, 1, received %d, %d, %d, %d, %d, %d\n", THFloatTensor_nDimension(offset), THFloatTensor_nDimension(topology), THFloatTensor_nDimension(xTable), THFloatTensor_nDimension(yTable), THFloatTensor_nDimension(zTable), THFloatTensor_nDimension(loss));
    return 0;
  }
  if (THFloatTensor_size(offset, 0)!=3 ){
    printf("Invalid shape!\n");
    printf("Expected 3xWxHxD, received %dx%dx%dx%d\n", C, W+1, H+1, D+1);
    return 0;
  }
  if (THFloatTensor_size(topology,0)!=(W*H*D)){
    printf("Invalid shape!\n");
    printf("Expected %dx%d, received %dx%d\n",  W*H*D, T, (int)THFloatTensor_size(topology, 0), T);
    return 0;
  }
  if (THFloatTensor_size(xTable,0)!=T || THFloatTensor_size(xTable,1)!=T ){
    printf("Invalid xTable!\n");
    return 0;
  }
  if (THFloatTensor_size(yTable,0)!=T || THFloatTensor_size(yTable,1)!=T ){
    printf("Invalid yTable!\n");
    return 0;
  }
  if (THFloatTensor_size(zTable,0)!=T || THFloatTensor_size(zTable,1)!=T ){
    printf("Invalid yTable!\n");
    return 0;
  }

  float accumu_loss = 0.0;

  // x direction
  for (int i=0; i<W-1; i++){
    for (int j=0; j<H; j++){
      for (int k=0; k<D; k++){
   
       float curr_loss = pairwise_loss(offset, topology, topology_empty, xTable, i, j, k, 0);
       accumu_loss = accumu_loss + curr_loss;
      }
    }
  }

  // y direction
  for (int i=0; i<W; i++){
    for (int j=0; j<H-1; j++){
      for (int k=0; k<D; k++){
   
       float curr_loss = pairwise_loss(offset, topology, topology_empty, yTable, i, j, k, 1);
       accumu_loss = accumu_loss + curr_loss;
      }
    }
  }

  // z direction
  for (int i=0; i<W; i++){
    for (int j=0; j<H; j++){
      for (int k=0; k<D-1; k++){
   
       float curr_loss = pairwise_loss(offset, topology, topology_empty, zTable, i, j, k, 2);
       accumu_loss = accumu_loss + curr_loss;
      }
    }
  }

  // inner cell 
  for (int i=0; i<W; i++){
    for (int j=0; j<H; j++){
      for (int k=0; k<D; k++){
   
       float curr_loss = pairwise_loss(offset, topology, topology_empty, innerTable, i, j, k, 3);
       accumu_loss = accumu_loss + curr_loss;
      }
    }
  }
  THFloatTensor_set1d(loss, 0, accumu_loss);
  
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
int curvature_constraint_backward(THFloatTensor *grad_output, THFloatTensor *offset, THFloatTensor *topology, THFloatTensor *topology_empty, THFloatTensor *xTable, THFloatTensor *yTable, THFloatTensor *zTable, THFloatTensor *innerTable, THFloatTensor *grad_offset ){
  long int W = THFloatTensor_size(offset, 1) - 1;
  long int H = THFloatTensor_size(offset, 2) - 1;
  long int D = THFloatTensor_size(offset, 3) - 1;

  THFloatTensor_zero(grad_offset);

  float grad_output_ = THFloatTensor_get1d(grad_output, 0);

  // x direction
  for (int i=0; i<W-1; i++){
    for (int j=0; j<H; j++){
      for (int k=0; k<D; k++){
         pairwise_grad(offset, topology, topology_empty, grad_offset, xTable, i, j, k, 0);
      }
    }
  }

  // y direction
  for (int i=0; i<W; i++){
    for (int j=0; j<H-1; j++){
      for (int k=0; k<D; k++){
         pairwise_grad(offset, topology, topology_empty, grad_offset, yTable, i, j, k, 1);
      }
    }
  }

  // z direction
  for (int i=0; i<W; i++){
    for (int j=0; j<H; j++){
      for (int k=0; k<D-1; k++){
         pairwise_grad(offset, topology, topology_empty, grad_offset, zTable, i, j, k, 2);
      }
    }
  }

  // inner cell 
  for (int i=0; i<W; i++){
    for (int j=0; j<H; j++){
      for (int k=0; k<D; k++){
         pairwise_grad(offset, topology, topology_empty, grad_offset, innerTable, i, j, k, 3);
      }
    }
  }
  THFloatTensor_mul(grad_offset, grad_offset, grad_output_);

 return 1;
}
