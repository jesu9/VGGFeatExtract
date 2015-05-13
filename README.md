# Python Wrapper for VGG Deep Feature Extraction

## Dependencies
- Python 2.7
- pycaffe
- cv2 (OpenCV Python wrapper)
- hdf5storage

## Features
- Wrappers to load images and videos with OpenCV
- Transform image based on VGG preprocessing
- Extract multi-layer feature embeddings with a single pass

## Quick Start
- See `video_demo.py` as an example to sample video frames, extract fc6/7 embedding features, and store features to MATLAB format.
- For USC HPCC users: Please `source /auto/iris-00/rn/chensun/iris_lib_setup.sh` to set up environment (anaconda, cv2, pycaffe, etc.)

## Contact
chensun@usc.edu