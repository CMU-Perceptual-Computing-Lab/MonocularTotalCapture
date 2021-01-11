#!/usr/bin/bash

set -e
# input param 
# $1: sequence name
# $2: data directory
seqName=$1
dataDir=$2
resName=$3
numFrame=$4

openposeDir=../openpose/

# convert to absolute path
MTCDir=$(readlink -f .)
dataDir=$(readlink -f $dataDir)
openposeDir=$(readlink -f $openposeDir)

cd $MTCDir
if [ -z "$numFrame" ]
then
	numFrame=$(ls $dataDir/$seqName/openpose_result/$seqName_* | wc -l)
fi

cd $MTCDir/FitAdam/
if [ ! -f ./build/viz_results ]; then
	echo "C++ project not correctly compiled. Please check your setting."
fi

# run visualization
./build/viz_results --root_dirs $dataDir --seqName $seqName --start 1 --end $((numFrame + 1)) --resName $resName
