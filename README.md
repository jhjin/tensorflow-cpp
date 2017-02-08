# Tensorflow C++ API example

The repository provides a basic image classification example using Tensorflow shared library (.so).
Tested on the Ubuntu 16.04 machine.


## Dependencies

Download `cudnn` library under the `lib` directory for CUDA.

```bash
lib/cudnn.h
lib/libcudnn.so.5  # with major version included in filename
```

and `make dependency` to install dependent packages via apt-get.
Update [CMakeLists.txt](CMakeLists.txt) according to your configuration if needed.


## Build and run

Calling the Makefile target will build tensorflow library,
download a pretrained model, and run the app.

```bash
make
```

If you need python interface, try `pip install lib/tensorflow*.whl`.
