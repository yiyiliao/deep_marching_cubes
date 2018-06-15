int occupancy_connectivity_forward( THFloatTensor *occupancy, THFloatTensor *loss );
int occupancy_connectivity_backward( THFloatTensor *grad_output, THFloatTensor *occupancy, THFloatTensor *grad_occupancy );
