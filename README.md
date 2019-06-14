# Monocular Total Capture
Code for CVPR19 paper "Monocular Total Capture: Posing Face, Body and Hands in the Wild"

# Dependencies
This code is tested on a Ubuntu 16.04 machine with a GTX 1080Ti GPU, with the following dependencies.
1. ffmpeg
2. Python 3.5 (with TensorFlow 1.5.0, OpenCV, Matplotlib)
3. cmake >= 2.8
4. OpenCV 2.4.13 (compiled with CUDA)
5. Ceres-Solver 1.13.0 (with SuiteSparse)
6. OpenGL, GLUT, GLEW
7. libigl <https://github.com/libigl/libigl>
8. wget

# Installation
1. git clone this repository; suppose the main directory is ${ROOT} on your local machine.
2. "cd ${ROOT}"
3. "bash download.sh"
4. Edit ${ROOT}/FitAdam/CMakeLists.txt, set line 13 to the "include" directory of libigl (this is a header only library).
5. "cd ${ROOT}/FitAdam/ && mkdir build && cd build"
6. "cmake .."
7. "make -j12"

# Usage
1. Suppose the video to be tested is named "${seqName}.mp4". Place it in "${ROOT}/${seqName}/${seqName}.mp4".
2. If the camera intrinsics is known, put it in "${ROOT}/${seqName}/calib.json" (refer to "POF/calib.json" for example).
3. In ${ROOT}, run "bash run_pipeline.sh ${seqName}"; if the subject in the video shows only upper body, run "bash run_pipeline.sh ${seqName} -f".

# Examples
"download.sh" automatically download 2 example videos to test. After successful installation run
```
bash run_pipeline.sh example_dance
```
or
```
bash run_pipeline.sh example_speech
```

Note: Facial expression of Adam model is unavailable to copyright issue.
