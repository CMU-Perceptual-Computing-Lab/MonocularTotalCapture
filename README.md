# Monocular Total Capture
Code for CVPR19 paper "Monocular Total Capture: Posing Face, Body and Hands in the Wild"

![Teaser Image](https://xiangdonglai.github.io/MTC_teaser.jpg)

Project website: [<http://domedb.perception.cs.cmu.edu/mtc.html>]

# Dependencies
This code is tested on a Ubuntu 16.04 machine with a GTX 1080Ti GPU, with the following dependencies.
1. ffmpeg
2. Python 3.5 (with TensorFlow 1.5.0, OpenCV, Matplotlib, packages installed with pip3)
3. cmake >= 2.8
4. OpenCV 2.4.13 (compiled from source with CUDA 9.0, CUDNN 7.0)
5. Ceres-Solver 1.13.0 (with SuiteSparse)
6. OpenGL, GLUT, GLEW
7. libigl <https://github.com/libigl/libigl>
8. wget
9. OpenPose

# Installation
1. git clone this repository; suppose the main directory is ${ROOT} on your local machine.
2. "cd ${ROOT}"
3. "bash download.sh"
4. git clone OpenPose <https://github.com/CMU-Perceptual-Computing-Lab/openpose> and compile. Suppose the main directory of OpenPose is ${openposeDir}, such that the compiled binary is at ${openposeDir}/build/examples/openpose/openpose.bin
5. Edit ${ROOT}/run_pipeline.sh: set line 13 to you ${openposeDir}
4. Edit ${ROOT}/FitAdam/CMakeLists.txt: set line 13 to the "include" directory of libigl (this is a header only library)
5. "cd ${ROOT}/FitAdam/ && mkdir build && cd build"
6. "cmake .."
7. "make -j12"

# Usage
1. Suppose the video to be tested is named "${seqName}.mp4". Place it in "${ROOT}/${seqName}/${seqName}.mp4".
2. If the camera intrinsics is known, put it in "${ROOT}/${seqName}/calib.json" (refer to "POF/calib.json" for example); otherwise, a default camera intrinsics will be used.
3. In ${ROOT}, run "bash run_pipeline.sh ${seqName}"; if the subject in the video shows only upper body, run "bash run_pipeline.sh ${seqName} -f".

# Examples
"download.sh" automatically download 2 example videos to test. After successful installation run
```
bash run_pipeline.sh example_dance
```
or
```
bash run_pipeline.sh example_speech -f
```

# License and Citation
This code can only be used for **non-commercial research purposes**. If you use this code in your research, please cite the following papers.
```
@inproceedings{xiang2019monocular,
  title={Monocular total capture: Posing face, body, and hands in the wild},
  author={Xiang, Donglai and Joo, Hanbyul and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}

@inproceedings{joo2018total,
  title={Total capture: A 3d deformation model for tracking faces, hands, and bodies},
  author={Joo, Hanbyul and Simon, Tomas and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

Some part of this code is modified from [lmb-freiburg/hand3d](https://github.com/lmb-freiburg/hand3d).

# Adam Model
We use the deformable human model [**Adam**](http://www.cs.cmu.edu/~hanbyulj/totalcapture/) in this code.

**The relationship between Adam and SMPL:** The body part of Adam is derived from [SMPL](http://smpl.is.tue.mpg.de/license_body) model by Loper et al. 2015. It follows SMPL's body joint hierarchy, but uses a different joint regressor. Adam does not contain the original SMPL model's shape and pose blendshapes, but uses its own version trained from Panoptic Studio database.

**The relationship between Adam and FaceWarehouse:** The face part of Adam is derived from [FaceWarehouse](http://kunzhou.net/zjugaps/facewarehouse/). In particular, the mesh topology of face of Adam is a modified version of the learned model from FaceWarehouse dataset. Adam does not contain the blendshapes of the original FaceWarehouse data, and facial expression of Adam model is unavailable due to copyright issues.

The Adam model is shared for research purpose only, and cannot be used for commercial purpose. Redistributing the original or modified version of Adam is also not allowed without permissions. 

# Special Notice
1. In our code, the output of ceres::AngleAxisToRotationMatrix is always a RowMajor matrix, while the function is designed for a ColMajor matrix. To account for this, please treat our output pose parameters as the opposite value. In other words, before exporting our pose parameter to other softwares, please multiply them by -1.
