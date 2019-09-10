#!/bin/bash
# $1 arg is the s3 path to the model.tar.gz model artifacts file.
mkdir tmp-fix
cd tmp-fix
aws s3 cp $1 .
gunzip model.tar.gz
echo Original contents:
tar xvf model.tar
rm model.tar
echo Updating inference.py and requirements.txt with your notebook copy
cp ../code/inference.py code/
cp ../code/requirements.txt code/
echo Updated contents:
tar cvf model.tar *
gzip model.tar
aws s3 cp model.tar.gz $1
cd ..
rm -rf tmp-fix
echo Done.