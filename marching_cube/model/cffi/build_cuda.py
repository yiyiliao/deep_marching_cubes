import os
import torch
import torch.utils.ffi

strBasepath = os.path.split(os.path.abspath(__file__))[0] + '/'

# distance
strHeaders = ['src/point_triangle_distance_cuda.h']
strSources = ['src/point_triangle_distance_cuda.c']
strDefines = [('WITH_CUDA', None)]
strObjects = ['src/point_triangle_distance_kernel.o']

# curvature constraint 
strHeaders_c = ['src/curvature_constraint_cuda.h']
strSources_c = ['src/curvature_constraint_cuda.c']
strDefines_c = [('WITH_CUDA', None)]
strObjects_c = ['src/curvature_constraint_kernel.o']

# grid_pooling 
strHeaders_g = ['src/grid_pooling_cuda.h', 'src/occupancy_to_topology_cuda.h']
strSources_g = ['src/grid_pooling_cuda.c', 'src/occupancy_to_topology_cuda.c']
strDefines_g = [('WITH_CUDA', None)]
strObjects_g = ['src/grid_pooling_kernel.o', 'src/occupancy_to_topology_kernel.o']

# occupancy connectivity 
strHeaders_o = ['src/occupancy_connectivity_cuda.h']
strSources_o = ['src/occupancy_connectivity_cuda.c']
strDefines_o = [('WITH_CUDA', None)]
strObjects_o = ['src/occupancy_connectivity_kernel.o']

ffi = torch.utils.ffi.create_extension(
    name='_ext.distance_cuda',
    headers=strHeaders,
    sources=strSources,
    verbose=False,
    with_cuda=any(strDefine[0] == 'WITH_CUDA' for strDefine in strDefines),
    package=False,
    relative_to=strBasepath,
    include_dirs=[os.path.expandvars('$CUDA_HOME') + '/include'],
    define_macros=strDefines,
    extra_objects=[os.path.join(strBasepath, strObject) for strObject in strObjects]
)

ffi_c = torch.utils.ffi.create_extension(
    name='_ext.curvature_cuda',
    headers=strHeaders_c,
    sources=strSources_c,
    verbose=False,
    with_cuda=any(strDefine[0] == 'WITH_CUDA' for strDefine in strDefines_c),
    package=False,
    relative_to=strBasepath,
    include_dirs=[os.path.expandvars('$CUDA_HOME') + '/include'],
    define_macros=strDefines_c,
    extra_objects=[os.path.join(strBasepath, strObject) for strObject in strObjects_c]
)
ffi_g = torch.utils.ffi.create_extension(
    name='_ext.forward_utils_cuda',
    headers=strHeaders_g,
    sources=strSources_g,
    verbose=False,
    with_cuda=any(strDefine[0] == 'WITH_CUDA' for strDefine in strDefines_g),
    package=False,
    relative_to=strBasepath,
    include_dirs=[os.path.expandvars('$CUDA_HOME') + '/include'],
    define_macros=strDefines_g,
    extra_objects=[os.path.join(strBasepath, strObject) for strObject in strObjects_g]
)

ffi_o = torch.utils.ffi.create_extension(
    name='_ext.occupancy_cuda',
    headers=strHeaders_o,
    sources=strSources_o,
    verbose=False,
    with_cuda=any(strDefine[0] == 'WITH_CUDA' for strDefine in strDefines_o),
    package=False,
    relative_to=strBasepath,
    include_dirs=[os.path.expandvars('$CUDA_HOME') + '/include'],
    define_macros=strDefines_o,
    extra_objects=[os.path.join(strBasepath, strObject) for strObject in strObjects_o]
)

if __name__ == '__main__':
    assert( torch.cuda.is_available() == True)
    ffi.build()
    ffi_c.build()
    ffi_g.build()
    ffi_o.build()
