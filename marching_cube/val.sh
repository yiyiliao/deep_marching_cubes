#!/bin/bash

set -e

ENCODER_TYPE=$1
DATA_DIR=$2
OUTPUT_DIR=$3
MODEL_FILE=$4

# step1: run the model in validation phase and get the off models as output
echo =================================
echo Runing the validation phase...
echo =================================


python val.py --encoder_type $ENCODER_TYPE \
	      --data_type shapenet \
       	      --num_voxels 32 \
	      --data_dir $DATA_DIR \
	      --output_dir $OUTPUT_DIR \
	      --model $MODEL_FILE \
	      --save_off 1


# step2: compare the distance between prediction and ground truth
echo =================================
echo Computing chamfer distance...
echo =================================
eval_bin=../tools/mesh-evaluation/bin
off_gt=${DATA_DIR}/mesh_shapenet_val
off_est=${OUTPUT_DIR}/mesh
eval_dir=${OUTPUT_DIR}/eval

mkdir -p ${eval_dir}

${eval_bin}/eval --input $off_est --reference $off_gt --output ${eval_dir} --n_points 3000

