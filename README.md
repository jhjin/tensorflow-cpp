# Tensorflow C++ API example

The repository provides a basic image classification example using Tensorflow shared library (.so).
This script automatically set up and build an environment for Tensorflow C++ API.
Tested on the Ubuntu 14.04 machine.

## Setup

clone this repo recursively.

```
git clone --recursive http://github.com/jhjin/tensorflow-cpp
```

then install dependent packages via apt-get.

```bash
./dependencies.sh
```

and manually download cuDNN for Linux 64bit from [NVIDIA](https://developer.nvidia.com/cudnn).
This script assumes that its header and library are in the following path.
(the version should be mentioned in the `so` filename)

```bash
mkdir -p include lib64
cp /path/to/cudnn.h include/cudnn.h
cp /path/to/libcudnn.so.5.1.3 lib64/libcudnn.so.5.1.3   # v5.1.3 for example
```

Now ready to run the script to build bazel (cmake-like tool from Google) and tensorflow.

```bash
./setup.sh
```

The build sometimes fails due to a download issue while fetching tensorflow internal dependencies.
Running the script again will simply solve this problem.

After build, generated header files are copied to include directories in a certain structure required by tensorflow.
Since the file copy is done quick and dirty without full understanding of the header structure,
this part is subject to break upon any update in tensorflow.


## Example

The image recognition demo is taken from
[tensorflow repo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image)
and used for an example.
Include `libtensorflow.so` in your library path and compile/run the app.

```bash
make
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib64 && ./app
```
