#!/usr/bin/bash
cd POF/
wget domedb.perception.cs.cmu.edu/data/mtc/mtc_snapshots.zip
unzip mtc_snapshots.zip
if [ ! -d snapshots ]; then
    echo "Pretrained model not extracted! Please check your setting."
else
    echo "Pretrained model successfully downloaded."
fi
cd ../FitAdam/model
wget domedb.perception.cs.cmu.edu/data/mtc/adam_blendshapes_348_delta_norm.json
cd ../include/
wget domedb.perception.cs.cmu.edu/data/mtc/InitializeAdamData.h
cd ../../data/
mkdir example_dance && cd example_dance
wget domedb.perception.cs.cmu.edu/data/mtc/example_dance.mp4
cd ../
mkdir example_speech && cd example_speech
wget domedb.perception.cs.cmu.edu/data/mtc/example_speech.mp4
