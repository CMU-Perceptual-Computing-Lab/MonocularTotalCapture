#!/usr/bin/bash
cd POF/
wget posefs1.perception.cs.cmu.edu/mtc/mtc_snapshots.zip
unzip mtc_snapshots.zip
if [ ! -d snapshots ]; then
    echo "Pretrained model not extracted! Please check your setting."
else
    echo "Pretrained model successfully downloaded."
fi
