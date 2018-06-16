#!/bin/bash

TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

echo nvcc -c -o src/point_triangle_distance_kernel.o src/point_triangle_distance_kernel.cu \
	--compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

nvcc -c -o src/point_triangle_distance_kernel.o src/point_triangle_distance_kernel.cu \
	--compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

echo nvcc -c -o src/curvature_constraint_kernel.o src/curvature_constraint_kernel.cu \
	--compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

nvcc -c -o src/curvature_constraint_kernel.o src/curvature_constraint_kernel.cu \
	--compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

echo nvcc -c -o src/grid_pooling_kernel.o src/grid_pooling_kernel.cu \
	--compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

nvcc -c -o src/grid_pooling_kernel.o src/grid_pooling_kernel.cu \
	--compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

echo nvcc -c -o src/occupancy_to_topology_kernel.o src/occupancy_to_topology_kernel.cu \
	--compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

nvcc -c -o src/occupancy_to_topology_kernel.o src/occupancy_to_topology_kernel.cu \
	--compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

echo nvcc -c -o src/occupancy_connectivity_kernel.o src/occupancy_connectivity_kernel.cu \
	--compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

nvcc -c -o src/occupancy_connectivity_kernel.o src/occupancy_connectivity_kernel.cu \
	--compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

python build_cuda.py
