import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['src/point_triangle_distance.c', 'src/commons.c']
headers = ['src/point_triangle_distance.h', 'src/commons.h']
sources_gridpool = ['src/grid_pooling.c', 'src/occupancy_to_topology.c', 'src/commons.c']
headers_gridpool = ['src/grid_pooling.h', 'src/occupancy_to_topology.h', 'src/commons.h']
sources_curvature = ['src/curvature_constraint.c', 'src/commons.c']
headers_curvature = ['src/curvature_constraint.h', 'src/commons.h']
sources_occupancy = ['src/occupancy_connectivity.c']
headers_occupancy = ['src/occupancy_connectivity.h']
sources_eval = ['src/pred_to_mesh.c', 'src/commons.c']
headers_eval = ['src/pred_to_mesh.h', 'src/commons.h']
defines = []
with_cuda = False

ffi = create_extension(
    '_ext.distance',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    extra_compile_args=["-std=c99"],
    with_cuda=with_cuda
)
ffi_g = create_extension(
    '_ext.forward_utils',
    headers=headers_gridpool,
    sources=sources_gridpool,
    define_macros=defines,
    relative_to=__file__,
    extra_compile_args=["-std=c99"],
    with_cuda=with_cuda
)
ffi_c = create_extension(
    '_ext.curvature',
    headers=headers_curvature,
    sources=sources_curvature,
    define_macros=defines,
    relative_to=__file__,
    extra_compile_args=["-std=c99"],
    with_cuda=with_cuda
)
ffi_o = create_extension(
    '_ext.occupancy',
    headers=headers_occupancy,
    sources=sources_occupancy,
    define_macros=defines,
    relative_to=__file__,
    extra_compile_args=["-std=c99"],
    with_cuda=with_cuda
)
ffi_eval = create_extension(
    '_ext.eval_util',
    headers=headers_eval,
    sources=sources_eval,
    define_macros=defines,
    relative_to=__file__,
    extra_compile_args=["-std=c99"],
    with_cuda=with_cuda
)


if __name__ == '__main__':
    ffi.build()
    ffi_g.build()
    ffi_c.build()
    ffi_o.build()
    ffi_eval.build()
