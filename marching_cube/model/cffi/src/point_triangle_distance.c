#include <TH/TH.h>
#include <stdio.h>
#include "commons.h"

const float eps=1e-8;
const float distance_empty=0.4;


/**
 * d_sqrdistance/d_x
 */
float d_sqrdistance_(float a, float b, float c, float d, float e, float f, float s, float t,
		float d_a, float d_b, float d_c, float d_d, float d_e, float d_f, float d_s, float d_t){
  return d_a*s*s + 2.0*a*d_s*s + 
	 d_c*t*t + 2.0*c*d_t*t + 
	 2.0*d_b*s*t + 2.0*b*d_s*t + 2*b*s*d_t +
	 2.0*d_s*d + 2.0*s*d_d + 
	 2.0*d_e*t + 2.0*e*d_t + d_f;  
}


/**
 * d_s/d_x
 */
float d_s_(float a, float b, float c, float d, float e, 
	float d_a, float d_b, float d_c, float d_d, float d_e,
       	float s_clamp, float t_clamp, float det){
  if (s_clamp==0) return 0;

  if (s_clamp+t_clamp<=1){
    float d_det = d_a*c + a*d_c - 2.0*b*d_b;
    float det2 = det*det;
    if (det2<eps) det2=eps;
    return ((d_b*e + b*d_e - d_c*d - c*d_d)*det -  (b*e-c*d)*d_det ) / ( det2 ); 
  }else if (s_clamp + t_clamp >1 && t_clamp > 0){
    float tmp = b*e - c*d + b*d - a*e;
    return ((d_b*e + b*d_e - d_c*d - c*d_d)*(b*d - a*e) - (b*e-c*d)*(d_b*d + b*d_d - d_a*e - a*d_e) ) / (tmp*tmp);
  }else{
    return 0;
  }
}


/**
 * d_t/d_x
 */
float d_t_(float a, float b, float c, float d, float e, 
	float d_a, float d_b, float d_c, float d_d, float d_e, 
       	float s_clamp, float t_clamp, float det){
  if (t_clamp==0) return 0;

  if (s_clamp+t_clamp<=1){
    float d_det = d_a*c + a*d_c - 2.0*b*d_b;
    float det2 = det*det;
    if (det2<eps) det2=eps;
    return ((d_b*d + b*d_d - d_a*e - a*d_e)*det -  (b*d-a*e)*d_det ) / ( det2 ); 
  }else if (s_clamp + t_clamp >1 && s_clamp > 0){
    float tmp = b*e - c*d + b*d - a*e;
    return ((d_b*d + b*d_d - d_a*e - a*d_e)*(b*e - c*d) - (b*d-a*e)*(d_b*e + b*d_e - d_c*d - c*d_d)) / (tmp * tmp);
  }else{
    return 0;
  }
}



/**
 * get the triangle vertices from the vertex displacement field
 * params:
 * 	offset 		vertex displacement field
 * 	i 		indice of the current cell on one of the directions
 * 	j 		indice of the current cell on one of the directions
 * 	k 		indice of the current cell on one of the directions
 * 	t 		indice of the topology type
 * return
 * 	triangle	vertices for all triangles in the specified topology type
 */
THFloatTensor* offset_to_triangles(THFloatTensor *offset, int i, int j, int k, int t){

  THFloatTensor *triangle = THFloatTensor_newWithSize3d( acceptTopology[1][t], 3, 3);
  float offset_c[3]={(float)i, (float)j, (float)k};

  THFloatTensor *vertices = offset_to_vertices(offset, i, j, k);

  for (int tri_ind = 0; tri_ind<acceptTopology[1][t]; tri_ind++){
    for (int vertex_ind = 0; vertex_ind<3; vertex_ind++){
	
	// copy the vertex coordinates to the triangle matrix
	int topology_ind = acceptTopology[0][t];
	for (int _i=0; _i<3; _i++){
	    THFloatTensor_set3d( triangle, tri_ind, _i, vertex_ind, THFloatTensor_get2d(vertices, _i, triTable[topology_ind][tri_ind*3+vertex_ind])+offset_c[_i] );
	}
    } 
  }

  THFloatTensor_free(vertices);
  return triangle;
}


/**
 * propagate the loss from the triangle vertices to the vertex displacement field
 * params:
 * 	grad_triangle 	input, gradient on the triangle vertices
 * 	grad_offset 	output, gradience on the vertex displacement field
 * 	i 		input, indice of the current cell on one of the directions
 * 	j 		input, indice of the current cell on one of the directions
 * 	k 		input, indice of the current cell on one of the directions
 * 	t 		input, indice of the topology type
 */
void grad_triangle_to_offset(THFloatTensor *grad_triangle, THFloatTensor *grad_offset, int i, int j, int k, int t){
  // for triangles in a single toplogy
  for (int tri_ind = 0; tri_ind<acceptTopology[1][t]; tri_ind++){
    // for vertices on the triangle
    for (int vertex_ind = 0; vertex_ind<3; vertex_ind++){

	// every vertex only contributes to the gradient of a single variable on the offset map
	int topology_ind = acceptTopology[0][t];
        int vertex = triTable[topology_ind][tri_ind*3+vertex_ind];

        float curr_grad = THFloatTensor_get4d( grad_offset, vertices_to_offset[vertex][0], 
  		      				     vertices_to_offset[vertex][1] + i,
  		                   		     vertices_to_offset[vertex][2] + j,
  						     vertices_to_offset[vertex][3] + k);
        THFloatTensor_set4d( grad_offset, vertices_to_offset[vertex][0], 
  		      		   vertices_to_offset[vertex][1] + i,
  		                   vertices_to_offset[vertex][2] + j, 
  				   vertices_to_offset[vertex][3] + k,
  				   curr_grad - THFloatTensor_get3d(grad_triangle, tri_ind, vertices_to_offset[vertex][0], vertex_ind ) );
	}
  }
}


/*
 * Forward function, calculating the distance from a set of points to one single triangle 
 * params: 
 * 	  triangle 	input, a single triangle, 3x3
 *  	  point 	input, points within the grid, Nx3
 *  	  distance  	output, distance from every point to the given triangle, N
 *
 */	
int point_triangle_distance_forward(THFloatTensor *triangle, THFloatTensor *point,
		       THFloatTensor *distance)
{
  // data format check
  if (THFloatTensor_nDimension(triangle)!=2 ||  THFloatTensor_nDimension(point)!=2 ||  THFloatTensor_nDimension(distance)!=1){
    printf("Invalid nDimension!\n");
    printf("Expected 2, 2, 2, received %d, %d, %d\n", THFloatTensor_nDimension(triangle), THFloatTensor_nDimension(point), THFloatTensor_nDimension(distance));
    return 0;
  }
  // triangle -> 3x3 
  if (THFloatTensor_size(triangle,0)!=3 || THFloatTensor_size(triangle,1)!=3){
    printf("Invalid shape of triangle!\n");
    return 0;
  }
  // point -> 1x3
  if (THFloatTensor_size(point,1)!=3){
    printf("Invalid shape of point!\n");
    return 0;
  }
  // distance -> 1
  if (THFloatTensor_size(point,0) != THFloatTensor_size(distance,0)){
    printf("Invalid shape of distance!\n");
    return 0;
  }
 
  float det, s, t, sqrdistance;

  // deep copy of a column
  THFloatTensor *_tmpB =  THFloatTensor_newSelect(triangle, 1, 0);
  THFloatTensor *_tmpE0 = THFloatTensor_newSelect(triangle, 1, 1);
  THFloatTensor *_tmpE1 = THFloatTensor_newSelect(triangle, 1, 2);
  THFloatTensor *B = THFloatTensor_newClone( _tmpB );
  THFloatTensor *E0 = THFloatTensor_newClone( _tmpE0 );
  THFloatTensor *E1 = THFloatTensor_newClone( _tmpE1 );
  THFloatTensor_free(_tmpB);
  THFloatTensor_free(_tmpE0);
  THFloatTensor_free(_tmpE1);

  THFloatTensor_csub(E0, E0, 1, B);
  THFloatTensor_csub(E1, E1, 1, B);

  int N = THFloatTensor_size(point,0);
  float a = THFloatTensor_dot(E0, E0);
  float b = THFloatTensor_dot(E0, E1);
  float c = THFloatTensor_dot(E1, E1);
  for (int i=0; i<N; i++){
    // deep copy
    THFloatTensor *_tmpD =  THFloatTensor_newSelect(point, 0, i);
    THFloatTensor *D = THFloatTensor_newClone( _tmpD );
    THFloatTensor_free(_tmpD);

    THFloatTensor_csub(D, B, 1, D);

    float d = THFloatTensor_dot(E0, D);
    float e = THFloatTensor_dot(E1, D);
    float f = THFloatTensor_dot(D, D);

    det = a*c - b*b;
    if (det<eps) det=eps;
    s = (b*e - c*d) / det;
    t = (b*d - a*e) / det;

    if (s<0) s=0;
    if (t<0) t=0;
    float norm = s+t;
    if (norm>1){
            s = s/norm;
            t = t/norm;
    }

    sqrdistance = s * ( a*s + b*t + 2.0*d ) + t * ( b*s + c*t + 2.0*e ) + f;
    THFloatTensor_set1d(distance, i, sqrdistance);
    THFloatTensor_free(D);
  }

  THFloatTensor_free(B);
  THFloatTensor_free(E0);
  THFloatTensor_free(E1);
  return 1;
}

/*
 * Backward function, calculating the gradient on a single triangle 
 * params: 
 * 	  grad_output   input, gradient on the output distance, 1
 * 	  triangle 	input, a single triangles, 3x3
 *  	  point 	input, points within the grid, Nx3
 *  	  grad_triangle output, gradient on the triangle, 3x3
 *  	  indices 	input, index indicating the nearest triangle, N
 *  	  triangle_ind 	input, indicating the index of given triangle among all triangles in the cell, 1
 *
 */	
int point_triangle_distance_backward(THFloatTensor *grad_output, THFloatTensor *triangle, THFloatTensor *point, 
		THFloatTensor *grad_triangle, THLongTensor *indices, int triangle_ind) 
{
  float t11, t12, t13, t21, t22, t23, t31, t32, t33;
  float p1, p2, p3;
  t11 = THFloatTensor_get2d(triangle, 0, 0);
  t12 = THFloatTensor_get2d(triangle, 0, 1);
  t13 = THFloatTensor_get2d(triangle, 0, 2);
  t21 = THFloatTensor_get2d(triangle, 1, 0);
  t22 = THFloatTensor_get2d(triangle, 1, 1);
  t23 = THFloatTensor_get2d(triangle, 1, 2);
  t31 = THFloatTensor_get2d(triangle, 2, 0);
  t32 = THFloatTensor_get2d(triangle, 2, 1);
  t33 = THFloatTensor_get2d(triangle, 2, 2);

  // TODO; find a better solution for deep copy a column
  THFloatTensor *_tmpB =  THFloatTensor_newSelect(triangle, 1, 0);
  THFloatTensor *_tmpE0 = THFloatTensor_newSelect(triangle, 1, 1);
  THFloatTensor *_tmpE1 = THFloatTensor_newSelect(triangle, 1, 2);
  THFloatTensor *B = THFloatTensor_newClone( _tmpB );
  THFloatTensor *E0 = THFloatTensor_newClone( _tmpE0 );
  THFloatTensor *E1 = THFloatTensor_newClone( _tmpE1 );
  THFloatTensor_free(_tmpB);
  THFloatTensor_free(_tmpE0);
  THFloatTensor_free(_tmpE1);

  THFloatTensor_csub(E0, E0, 1, B);
  THFloatTensor_csub(E1, E1, 1, B);


  int N = THFloatTensor_size(point,0);
  float a = THFloatTensor_dot(E0, E0);
  float b = THFloatTensor_dot(E0, E1);
  float c = THFloatTensor_dot(E1, E1);
  float d_t11,d_t21,d_t31,d_t12,d_t22,d_t32,d_t13,d_t23,d_t33;
  d_t11=d_t21=d_t31=d_t12=d_t22=d_t32=d_t13=d_t23=d_t33 = 0;

  int curr_ind;

  // all points has the same contributes in one single grid
  float grad_output_ = THFloatTensor_get1d(grad_output, 0);

  for (int i=0; i<N; i++){
    
    // only backpropagate to the nearest triangle
    curr_ind =  THLongTensor_get1d(indices, i);
    if (curr_ind!=triangle_ind) continue; 
    
    p1 = THFloatTensor_get2d(point, i, 0);
    p2 = THFloatTensor_get2d(point, i, 1);
    p3 = THFloatTensor_get2d(point, i, 2);

    // TODO; find a better solution for deep copy
    THFloatTensor *_tmpD =  THFloatTensor_newSelect(point, 0, i);
    THFloatTensor *D = THFloatTensor_newClone( _tmpD );
    THFloatTensor_free(_tmpD);

    THFloatTensor_csub(D, B, 1, D);
    
    float d = THFloatTensor_dot(E0, D);
    float e = THFloatTensor_dot(E1, D);
    float f = THFloatTensor_dot(D, D);

    float det, s, t;
    det = a*c - b*b;
    if (det<eps) det=eps;
    s = (b*e - c*d) / det;
    t = (b*d - a*e) / det;
  

    float d_a,d_b,d_c,d_d,d_e,d_f;
    float s_clamp = s; 
    float t_clamp = t;
    if (s<0) s_clamp=0;
    if (t<0) t_clamp=0;
    float s_norm = s_clamp;
    float t_norm = t_clamp;
    float norm = s_clamp+t_clamp;
    if (norm>1){
            s_norm = s_clamp/norm;
            t_norm = t_clamp/norm;
    }
    // t11
    d_a = 2*t11 - 2*t12; d_b = 2*t11 - t12 - t13; d_c = 2*t11 - 2*t13; d_d = p1 - 2*t11 + t12; d_e = p1 - 2*t11 + t13; d_f = 2*t11 - 2*p1; 
    d_t11 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));
    // t21
    d_a = 2*t21 - 2*t22; d_b = 2*t21 - t22 - t23; d_c = 2*t21 - 2*t23; d_d = p2 - 2*t21 + t22; d_e = p2 - 2*t21 + t23; d_f = 2*t21 - 2*p2; 
    d_t21 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));
    // t31
    d_a = 2*t31 - 2*t32; d_b = 2*t31 - t32 - t33; d_c = 2*t31 - 2*t33; d_d = p3 - 2*t31 + t32; d_e = p3 - 2*t31 + t33; d_f = 2*t31 - 2*p3; 
    d_t31 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));

    // t12
    d_a = 2*t12 - 2*t11; d_b = t13 - t11; d_c = 0.0; d_d = t11 - p1; d_e = 0.0; d_f = 0.0; 
    d_t12 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));
    // t22
    d_a = 2*t22 - 2*t21; d_b = t23 - t21; d_c = 0.0; d_d = t21 - p2; d_e = 0.0; d_f = 0.0; 
    d_t22 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));
    // t32
    d_a = 2*t32 - 2*t31; d_b = t33 - t31; d_c = 0.0; d_d = t31 - p3; d_e = 0.0; d_f = 0.0; 
    d_t32 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));

    // t13
    d_a = 0.0; d_b = t12 - t11; d_c = 2*t13 - 2*t11; d_d = 0.0; d_e = t11 - p1; d_f = 0.0; 
    d_t13 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));
    // t23
    d_a = 0.0; d_b = t22 - t21; d_c = 2*t23 - 2*t21; d_d = 0.0; d_e = t21 - p2; d_f = 0.0; 
    d_t23 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));
    // t33
    d_a = 0.0; d_b = t32 - t31; d_c = 2*t33 - 2*t31; d_d = 0.0; d_e = t31 - p3; d_f = 0.0; 
    d_t33 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));

    THFloatTensor_free(D);
  }

  THFloatTensor_set2d(grad_triangle, 0, 0 , d_t11);
  THFloatTensor_set2d(grad_triangle, 1, 0 , d_t21);
  THFloatTensor_set2d(grad_triangle, 2, 0 , d_t31);

  THFloatTensor_set2d(grad_triangle, 0, 1 , d_t12);
  THFloatTensor_set2d(grad_triangle, 1, 1 , d_t22);
  THFloatTensor_set2d(grad_triangle, 2, 1 , d_t32);

  THFloatTensor_set2d(grad_triangle, 0, 2 , d_t13);
  THFloatTensor_set2d(grad_triangle, 1, 2 , d_t23);
  THFloatTensor_set2d(grad_triangle, 2, 2 , d_t33);

  THFloatTensor_free(B);
  THFloatTensor_free(E0);
  THFloatTensor_free(E1);
  return 1;
}


/*
 * Forward function, calculating the distance from a set of points to one single topology
 * params: 
 * 	  triangles 	input, triangles within a single topology, Tx3x3
 *  	  point 	input, points within the grid, Nx3
 *  	  distance  	output, distance from every point to the nearest triangle, N
 *  	  indices 	output, index indicating the nearest triangle, saved for back propagation, N
 *
 */	
int point_mesh_distance_forward(THFloatTensor *triangles, THFloatTensor *point,
		       THFloatTensor *distance, THLongTensor *indices)
{
  // data format check
  if (THFloatTensor_nDimension(triangles)!=3 ||  THFloatTensor_nDimension(point)!=2 ||  THFloatTensor_nDimension(distance)!=1){
    printf("Invalid nDimension!\n");
    printf("Expected 3, 2, 2, received %d, %d, %d\n", THFloatTensor_nDimension(triangles), THFloatTensor_nDimension(point), THFloatTensor_nDimension(distance));
    return 0;
  }
  // triangles -> Mx3x3 
  if (THFloatTensor_size(triangles,1)!=3 || THFloatTensor_size(triangles,2)!=3){
    printf("Invalid shape of triangle!\n");
    return 0;
  }
  // point -> Nx3
  if (THFloatTensor_size(point,1)!=3){
    printf("Invalid shape of point!\n");
    return 0;
  }
  // distance -> N 
  if (THFloatTensor_size(point,0) != THFloatTensor_size(distance,0)){
    printf("Invalid shape of distance!\n");
    return 0;
  }
  
  // calculate the distance from every point to every triangle
  THFloatTensor *distances = THFloatTensor_newWithSize2d( THFloatTensor_size(distance, 0),  THFloatTensor_size(triangles,0) );
  for (int i=0; i<THFloatTensor_size(triangles,0); i++){
    THFloatTensor *triangle_single = THFloatTensor_newSelect(triangles, 0, i);
    THFloatTensor *distance_single = THFloatTensor_newSelect(distances, 1, i);

    point_triangle_distance_forward( triangle_single, point, distance_single);

    THFloatTensor_free(triangle_single);
    THFloatTensor_free(distance_single);
  }
  // take the min value and save the indices for backward
  THFloatTensor_min( distance, indices, distances, 1, 1);

  THLongTensor_squeeze1d( indices, indices, 1 );
  THFloatTensor_squeeze1d( distance, distance, 1 );

  THFloatTensor_free(distances);

  return 1;
}

/*
 * Backward function, calculating gradient for every triangle 
 * params: 
 * 	  grad_output   input, gradient on the output distance, 1
 * 	  triangles 	input, triangles within a single topology, Tx3x3
 *  	  point 	input, points within the grid, Nx3
 *  	  grad_triangle output, gradient on every triangle, Tx3x3
 *  	  indices 	input, index indicating the nearest triangle, N
 *
 */	
int point_mesh_distance_backward(THFloatTensor *grad_output, THFloatTensor *triangles, THFloatTensor *point, 
		THFloatTensor *grad_triangles, THLongTensor *indices) 
{
  for (int i=0; i<THFloatTensor_size(triangles,0); i++){
    THFloatTensor *triangle_single = THFloatTensor_newSelect(triangles, 0, i);
    THFloatTensor *grad_triangle_single = THFloatTensor_newSelect(grad_triangles, 0, i);

    point_triangle_distance_backward(grad_output, triangle_single, point, grad_triangle_single, indices, i );

    THFloatTensor_free(triangle_single);
    THFloatTensor_free(grad_triangle_single);
  }
  
  return 1;

}


/*
 * Forward function, calculating the point to mesh distances for all grids
 * params: 
 * 	  offset 	input, offset map for x,y,z directions, 3x(W+1)x(H+1)x(D+1) 
 *  	  points 	input, all points, N_allx3
 *  	  distances  	output, point to mesh distances for every grid for every topolopy, (WxHxD)xT 
 *  	  indices_all   output, to record which triangle in each topology is the nearest one for backpropagation, N_allxT
 *
 */	
int point_topology_distance_forward(THFloatTensor *offset, THFloatTensor *points, THFloatTensor *distances, THLongTensor *indices_all){
  // data format check
  if (THFloatTensor_nDimension(offset)!=4 ||  THFloatTensor_nDimension(points)!=2 || THFloatTensor_nDimension(distances)!=2){
    printf("Invalid nDimension!\n");
    printf("Expected 4, 2, 2, received %d, %d, %d \n", THFloatTensor_nDimension(offset), THFloatTensor_nDimension(points), THFloatTensor_nDimension(distances));
    return 0;
  }
  int W,H,D,T;
  W = THFloatTensor_size(offset,1)-1; 
  H = THFloatTensor_size(offset,2)-1; 
  D = THFloatTensor_size(offset,3)-1; 
  T = THFloatTensor_size(distances, 1);

  // record which triangle the point is assigned to
  THLongTensor_fill(indices_all, -1);

  for (int i=0; i<W; i++){
    for (int j=0; j<H; j++){
      for (int k=0; k<D; k++){
	int ind = i*(H*D) + j*D + k;
        THLongTensor *point_indices = points_in_grid(points, i, j, k);
	// if the grid is empty, directly assign constant values  
	if (THLongTensor_nDimension(point_indices)==0){
	  for (int t=0; t<T-1; t++){
	    THFloatTensor_set2d( distances, ind, t, distance_empty );
	  }
	  THFloatTensor_set2d( distances, ind, T-1, 0.0 );
	// if the grid is not empty 
	}else{ 
	
	  THFloatTensor *distance = THFloatTensor_newWithSize1d( THLongTensor_size(point_indices, 0) );
	  THLongTensor *indices = THLongTensor_newWithSize1d( THLongTensor_size(point_indices, 0) );


	  for (int t=0; t<T-1; t++){
            THFloatTensor *points_grid = THFloatTensor_new();
	    THFloatTensor_indexSelect(points_grid, points, 0, point_indices);
	    THFloatTensor *triangles = offset_to_triangles(offset, i, j, k, t);
	    // allocate distance and indices

	    point_mesh_distance_forward(triangles, points_grid, distance, indices);

	    THFloatTensor_set2d( distances, ind, t, THFloatTensor_meanall(distance) );

	    // select a single column of indices_all and copy the value
	    THLongTensor *indices_t = THLongTensor_newSelect(indices_all, 1, t); 
	    THLongTensor_indexCopy(indices_t, 0, point_indices, indices);

	    THLongTensor_free(indices_t);
	    THFloatTensor_free(triangles);
	    THFloatTensor_free(points_grid);
	  }

	  THFloatTensor * _tmp_distance_row = THFloatTensor_newSelect(distances, 0, ind);
	  THFloatTensor_set2d( distances, ind, T-1, THFloatTensor_maxall(_tmp_distance_row) * 10.0 );
	  THFloatTensor_free(_tmp_distance_row);

	  THLongTensor_free(indices);
	  THFloatTensor_free(distance);

	}
	THLongTensor_free(point_indices);

      }
    }
  }
  return 1;
  
}

/*
 * Backward function, calculating the gradients for the full offset map 
 * params: 
 *  	  grad_output   input, gradient on the output distances, (WxHxD)xT
 * 	  offset 	input, offset map for x,y,z directions, 3x(W+1)x(H+1)x(D+1) 
 *  	  points 	input, all points, N_allx3
 *  	  indices_all   input, recorded which triangle in each topology is the nearest one for backpropagation, N_allxT
 *  	  grad_offset  	output, gradient on the full offset map, 3x(W+1)x(H+1)x(D+1)  
 *
 */	
int point_topology_distance_backward(THFloatTensor *grad_output, THFloatTensor *offset, THFloatTensor *points, THLongTensor *indices_all, THFloatTensor *grad_offset){

  int W,H,D,T;
  W = THFloatTensor_size(offset,1)-1; 
  H = THFloatTensor_size(offset,2)-1; 
  D = THFloatTensor_size(offset,3)-1; 
  T = THFloatTensor_size(grad_output, 1);


  for (int i=0; i<W; i++){
    for (int j=0; j<H; j++){
      for (int k=0; k<D; k++){
	int ind = i*(H*D) + j*D + k;
	// no gradient if the grid is empty  
        THLongTensor *point_indices = points_in_grid(points, i, j, k);
	if (THLongTensor_nDimension(point_indices)==0) {
	  THLongTensor_free(point_indices);
	  continue;
	}


	for (int t=0; t<T-1; t++){
          THFloatTensor *points_grid = THFloatTensor_new();
	  THFloatTensor_indexSelect(points_grid, points, 0, point_indices);

	  THFloatTensor *triangles = offset_to_triangles(offset, i, j, k, t);
	
	  THLongTensor *indices_t = THLongTensor_newSelect(indices_all, 1, t); 
	  THLongTensor *indices = THLongTensor_new();
	  THLongTensor_indexSelect(indices, indices_t, 0, point_indices);


	  float element = THFloatTensor_get2d(grad_output, ind, t);
	  THFloatTensor *grad_output_element = THFloatTensor_newWithSize1d(1);
	  THFloatTensor_fill(grad_output_element, element / (float)THLongTensor_size(point_indices, 0));

	  THFloatTensor *grad_triangles = THFloatTensor_newWithSize3d (THFloatTensor_size(triangles,0), THFloatTensor_size(triangles, 1), THFloatTensor_size(triangles, 2));
	  THFloatTensor_fill(grad_triangles, 0);
          point_mesh_distance_backward(grad_output_element, triangles, points_grid, grad_triangles, indices); //THFloatTensor *grad_point,

	  grad_triangle_to_offset(grad_triangles, grad_offset, i, j, k, t);

	  
	  THFloatTensor_free(grad_output_element);
	  THFloatTensor_free(grad_triangles);
	  THLongTensor_free(indices);
	  THLongTensor_free(indices_t);
	  THFloatTensor_free(triangles);
	  THFloatTensor_free(points_grid);
	}

	THLongTensor_free(point_indices);
      }
    }
  }
  return 1;
}
