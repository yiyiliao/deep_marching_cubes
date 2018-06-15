# !/bin/bash

ENCODER_TYPE=$1
DATA_DIR=$2
OUTPUT_DIR=$3

python train.py --encoder_type $ENCODER_TYPE \
		--data_type shapenet \
       		--num_voxels 32 \
		--data_dir $DATA_DIR \
	    	--output_dir $OUTPUT_DIR

