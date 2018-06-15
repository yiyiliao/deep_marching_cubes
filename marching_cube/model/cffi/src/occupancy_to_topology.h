int occupancy_to_topology_forward( THFloatTensor *occupancy, THFloatTensor *topology );
int occupancy_to_topology_backward( THFloatTensor *grad_output, THFloatTensor *occupancy, THFloatTensor *topology, THFloatTensor *grad_occupancy );
