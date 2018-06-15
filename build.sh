#!/bin/bash

set -e

echo ================================
echo Builing 3D c extensions...
echo ================================
cd dmc/model/cffi
python build.py

echo ================================
echo Builing 3D cuda extensions...
echo ================================
./build_cuda.sh

cd ../../../

echo Done!
