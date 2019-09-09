#!/bin/bash
# $1 arg is the s3 path to the model.tar.gz model artifacts file.
mkdir tmp-fix
cd tmp-fix
aws s3 cp $1 .
gunzip model.tar.gz
tar xvf model.tar
rm model.tar
cp ../code/* code/
tar cvf model.tar *
gzip model.tar
aws s3 cp model.tar.gz $1
cd ..
rm -rf tmp-fix