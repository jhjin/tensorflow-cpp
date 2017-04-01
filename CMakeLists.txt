cmake_minimum_required(VERSION 2.8.3)
project(tensorflow)
include(ExternalProject)
find_package(PythonInterp 2.7)
find_package(PythonLibs 2.7)
get_filename_component(PYTHON_LIBRARIES_DIR ${PYTHON_LIBRARIES} PATH)

find_package(CUDA)
if(CUDA_FOUND)
  set(TF_NEED_CUDA "1")
  set(TF_BUILD_CUDA "--config=cuda")
  set(CUDNN_INSTALL_PATH ${CMAKE_CURRENT_BINARY_DIR})
  set(CUDNN_VERSION "5")
  file(COPY ${CMAKE_SOURCE_DIR}/lib/cudnn.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  file(COPY ${CMAKE_SOURCE_DIR}/lib/libcudnn.so.${CUDNN_VERSION} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
else()
  set(TF_NEED_CUDA "0")
endif()

ExternalProject_Add(${PROJECT_NAME}-ext
  PREFIX ext
  GIT_REPOSITORY http://github.com/tensorflow/tensorflow.git
  GIT_TAG 1044c0ee945327294d2d6fdd93f6b4da9f529cf7
  BUILD_IN_SOURCE 1
  INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}

  CONFIGURE_COMMAND GCC_HOST_COMPILER_PATH=${CMAKE_C_COMPILER}
                    PYTHON_BIN_PATH=${PYTHON_EXECUTABLE}
                    PYTHON_LIB_PATH=${PYTHON_LIBRARIES_DIR}
                    CUDA_TOOLKIT_PATH=${CUDA_TOOLKIT_ROOT_DIR}
                    CUDNN_INSTALL_PATH=${CUDNN_INSTALL_PATH}
                    CC_OPT_FLAGS="-march=native"
                    TF_ENABLE_XLA=0
                    TF_NEED_JEMALLOC=1
                    TF_NEED_HDFS=0
                    TF_NEED_GCP=0
                    TF_NEED_OPENCL=0
                    TF_NEED_CUDA=${TF_NEED_CUDA}
                    TF_CUDA_CLANG=0
                    TF_CUDA_VERSION=${CUDA_VERSION}
                    TF_CUDNN_VERSION=${CUDNN_VERSION}
                    TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2
                    ./configure

  BUILD_COMMAND bazel build -c opt -j 200 --config=opt ${TF_BUILD_CUDA} //tensorflow:libtensorflow_cc.so
                                                                        //tensorflow/tools/pip_package:build_pip_package
  INSTALL_COMMAND cp <SOURCE_DIR>/bazel-bin/tensorflow/libtensorflow_cc.so ${CMAKE_SOURCE_DIR}/lib/libtensorflow.so
          COMMAND <SOURCE_DIR>/bazel-bin/tensorflow/tools/pip_package/build_pip_package ${CMAKE_SOURCE_DIR}/lib
	  COMMAND /bin/bash -c "wheel unpack ${CMAKE_SOURCE_DIR}/lib/tensorflow-*.whl"
	  COMMAND /bin/bash -c "rsync -am <SOURCE_DIR>/tensorflow-*/tensorflow-*.data/purelib/tensorflow/include ${CMAKE_SOURCE_DIR}"
	  COMMAND /bin/bash -c "rsync -am --include='*.h' -f 'hide,! */' <SOURCE_DIR>/bazel-genfiles/tensorflow ${CMAKE_SOURCE_DIR}/include"
	  COMMAND /bin/bash -c "rsync -am --include='*.h' -f 'hide,! */' <SOURCE_DIR>/tensorflow ${CMAKE_SOURCE_DIR}/include"
)
