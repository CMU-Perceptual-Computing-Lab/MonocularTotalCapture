# This is a fork...
of the original Monocular Total Capture code for use in the [3D Physical Mocap project](https://git.corp.adobe.com/adobe-research/3d-contact-pose). 

The original README is below with dependencies and installation instructions. All the versions they specify below are a big pain to get to work together, but as far as I can tell it's the only way to get the code working. Here are some suggestions in getting it set up properly from what I can remember:

## Installation
* First make sure you have CUDA 9.0 and CUDNN 7.0 installed.
* Next build OpenCV. Download the source for the version specified below (2.4.13). In order to get this version to build with CUDA 9.0 you need to patch the source with the changed specified [in this repository](https://github.com/davidstutz/opencv-2.4-cuda-9-patch). I also remember having some difficulties with the gcc version, so if you're running into issues, consider changing the version of gcc/g++ you use. The exact command I used to build was: `cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_CXX_COMPILER=/usr/bin/g++-6 -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -DUSE_CUDA=ON -DCUDA_GENERATION=Pascal -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..`. 
* Next build Caffe (do this separately, do not build it with OpenPose). Specifically, clone the Caffe repo and then revert to [this commit](https://github.com/BVLC/caffe/commit/f019d0dfe86f49d1140961f8c7dec22130c83154) and build as instructed in the [documentation for building with cmake](https://caffe.berkeleyvision.org/installation.html). This is necessary to avoid errors later when building OpenPose.
    * If you have trouble building while linking with OpenCV (i.e. get an error like ` /usr/bin/x86_64-linux-gnu-ld: cannot find -lopencv_dep_nppial ` you may have to softlink the CUDA libraries that it's looking for so that it can find it. For the error above, this would be `ln -s /usr/local/cuda/libnppial.so /usr/lib/libopencv_dep_nppial`. It's the exact same for other libraries that it throws this error for. 
* Next build OpenPose. Make sure to point it to the custom Caffe installation. The exact build instruction I used was `cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++-6 -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -DBUILD_CAFFE=OFF -DCaffe_INCLUDE_DIRS=/home/rempe/projects/caffe/build/install/include -DCaffe_LIBS=/home/rempe/projects/caffe/build/lib/libcaffe.so -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..`. Again beware of gcc/g++ versions.
* Make sure you have all other dependencies listed below installed and follow the installation instructions. 
* Finally, build monocular total capture. The exact build command I used was `cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++-6 -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -DGFLAGS_INCLUDE_DIR=/usr/include/gflags ..`. If you get some weird errors about gflags, run the exact same command again and it might go away.

## Notable Changes
* New [program to visualize total capture results](./FitAdam/viz_results.cpp).
* New [program to process total capture results](./FitAdam/process_results.cpp) and write them out to be used as initialization in the physical mocap optimization pipeline. 
* Scripts to run both the [visualization](./run_visualization.sh) and [processing](./run_processing.sh) and a modified version of the script to [run the total capture pipeline](./run_pipeline_no_ffmpeg.sh) that assumes each image frame has already been extracted from the video. All these scripts are called from the [python script](https://git.corp.adobe.com/adobe-research/3d-contact-pose/blob/master/scripts/run_totalcap.py) that runs total capture, visualizes the results, and does all processing which is provided in the physical mocap repo. This is the recommended way to run the pipeline; please see that repo for instructions to do this and what files need to be copied over for use in the physical mocap pipeline.

# ------------- ORIGINAL BELOW HERE -----------------

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
