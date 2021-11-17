# UQ Thesis Project (METR4901 2018-2019): Online Action Detection and Multi-Task Predictions with Kinect and Deep Learning

Demo video: https://www.youtube.com/watch?v=U1ZkbBfO4SY

## Dependencies
- Hardware: 
    * Microsoft Kinect v1 for XBox 360 / Windows
    * NVIDIA GPU (for training)
- OS: 
    * Windows 10 1803 or 1809, if using the current GUIs
    * Windows / Linux / Mac OS should all support neural network training and evaluations
- Utility: 
    * GPU driver (for training)
    * CUDA 9.0 (for training)
    * cuDNN 7.3.1 (for training)
    * Kinect SDK v1.8 (Windows only)
    * Microsoft Visual Studio 2015 incl. Python Development Tools (optional, to compile PyTorch C++/CUDA kernels in Windows)
    * OpenNI 2.2 and NiTE 2 (optional, if using PrimeSense for Kinect back end, supports Linux but only 15-joint skeleton, not currently used) 
    * Qt (optional, if using Qt Designer for UI design)
- Programming Language: Python 3.6, Anaconda
- Software Packages: Please see ```requirements.txt```.
- Datasets in their original formats (Optional): 
    * G3D: http://dipersec.king.ac.uk/G3D/
    * OAD: http://www.icst.pku.edu.cn/struct/Projects/OAD.html
    * PKU-MMD: http://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html
    * K3Da: https://filestore.leightley.com/k3da/

## Overview
* ```dataset/*``` contains ADT/CDT for skeleton datasets.
* ```demo/*``` contains trained weights for demo. Do NOT delete even if executable has been compiled.
* ```network/*``` contains neural network and training code.
* ```NiTE2/*``` and ```OpenNI2/*``` are PrimeSense driver and skeleton tracking binaries. They can be neglected if not using PrimeSense back end for Kinect.
* ```pykinect/*``` is the Python3-only binding for Kinect SDK v1.8. This is not the same as the PyKinect on pypi which only supports Python 2.7.
* ```sensor/*``` contains ADT/CDT for RGB-D sensors.
* ```sru/*``` contains a modified pypi SRU (which by default doesn't compile kernels correctly in Windows) and prebuilt CPU kernel binaries for Windows.
* ```trained``` is the directory to store instantaneous trained weights.
* ```ui/*``` contains code for GUIs.
* ```utils/*``` contains utility functions such as model evaluation, data processing, etc.
* ```validation``` is a temporary directory for batch-evaluation of trained models.
* ```global_configs.py``` stores constants and default settings of the entire software project.
* ```action_predictor.py``` runs the demo GUI.
* ```dataset_collector.py``` runs the dataset collector GUI.
* ```train_jcm.py``` trains models for action and attribute predictions.
* ```train_jcr.py``` trains models for action detections and forecasts.
* ```ui2py.ps1``` converts raw frontend code ```*.ui``` to ```*.py```.
* ```build_exe.ps1``` rebuilds the executable ```action_predictor.exe```.
* ```serialize_dataset.py``` serializes raw datasets so that training can be sped up.

## Additional Notes
- PrimeSense back end for Kinect is not (but used to be) supported in the GUIs.
- If one wishes to run GUIs on Linux, implement 2nd skeleton tracking for ```KinectNI``` in ```sensor/kinect.py```, as well as train models on 15-joint skeleton data.
- GUIs and the training modules are independent of each other. The former should not need GPUs to execute.
