#!/bin/bash
# $1 arg is the s3 path to the model.tar.gz model artifacts file.
mkdir tmp-fix
cd tmp-fix
aws s3 cp $1/model.tar.gz .
gunzip model.tar.gz
tar xvf model.tar
rm model.tar
cp ../code/requirements-gpu.txt code/requirements.txt
tar cvf model_gpu.tar *
gzip model_gpu.tar
aws s3 cp model_gpu.tar.gz $1/
cd ..
rm -rf tmp-fix