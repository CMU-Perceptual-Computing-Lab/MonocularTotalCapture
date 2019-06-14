#!/usr/bin/bash
cd POF/
wget posefs1.perception.cs.cmu.edu/mtc/mtc_snapshots.zip
unzip mtc_snapshots.zip
if [ ! -d snapshots ]; then
    echo "Pretrained model not extracted! Please check your setting."
else
    echo "Pretrained model successfully downloaded."
fi
cd ../FitAdam/model
wget posefs1.perception.cs.cmu.edu/mtc/adam_blendshapes_348_delta_norm.json
cd ../include/
wget posefs1.perception.cs.cmu.edu/mtc/InitializeAdamData.h
cd ../../data/
mkdir example_dance && cd example_dance
wget posefs1.perception.cs.cmu.edu/mtc/example_dance.mp4
cd ../
mkdir example_speech && cd example_speech
wget posefs1.perception.cs.cmu.edu/mtc/example_speech.mp4
