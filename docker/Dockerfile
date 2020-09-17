FROM nvidia/cuda:9.0-devel

# Ensure that your default-runtime has CUDA support during compilation, i.e.:
# {
#   "runtimes": {
#     "nvidia": {
#       "path": "nvidia-container-runtime",
#       "runtimeArgs": []
#     }
#   },
#   "default-runtime": "nvidia"
# }
#
# Build the docker image:
# $ docker build  . -t marching_cubes
#
# Using $DATA to point to the ShapeNet data, invoke:
# $ docker run -v $DATA:/data/ -it marching_cubes:latest
#
# Follow the instructions for training/validation

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git vim python-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install cffi matplotlib scipy
RUN pip install torch==0.3.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN cd /home && git clone https://github.com/yiyiliao/deep_marching_cubes.git
RUN cd /home/deep_marching_cubes && ./build.sh
