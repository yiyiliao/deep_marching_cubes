int grid_pooling_forward( THFloatTensor *point, THFloatTensor *feat_points, THLongTensor *shape, THFloatTensor *feat_cell, THLongTensor *indices);
int grid_pooling_backward( THFloatTensor *grad_output, THLongTensor *indices, THFloatTensor *grad_feat_points);
