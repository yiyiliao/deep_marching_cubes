int point_triangle_distance_forward(THFloatTensor *triangle, THFloatTensor *point,
		       THFloatTensor *distance);
int point_triangle_distance_backward(THFloatTensor *grad_output, THFloatTensor *triangle, THFloatTensor *point,
	          	THFloatTensor *grad_triangle); 
int point_mesh_distance_forward(THFloatTensor *triangles, THFloatTensor *point,
		       THFloatTensor *distance, THLongTensor *indices);
int point_mesh_distance_backward(THFloatTensor *grad_output, THFloatTensor *triangles, THFloatTensor *point, 
		THFloatTensor *grad_triangle, THLongTensor *indices); 
int point_topology_distance_forward(THFloatTensor *offset, THFloatTensor *points, THFloatTensor *distances, THLongTensor *indices_all);
int point_topology_distance_backward(THFloatTensor *grad_output, THFloatTensor *offset, THFloatTensor *points, THLongTensor *indices_all, THFloatTensor *grad_offset);
