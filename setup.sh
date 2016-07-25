#!/usr/bin/env bash

wget_and_check() {
  local file_name=$1
  local file_url=$2
  local file_md5=$3

  if [ ! -f $file_name ]; then
    mkdir -p $(dirname $file_name)
    wget -O $file_name $file_url --no-check-certificate
  fi

  if [ "$(md5sum $file_name | awk '{print $1}')" != "$file_md5" ]; then
    rm -f $file_name
    echo "==> File corrupted. Attempt to download it again.."
    wget_and_check $file_name $file_url $file_md5;
  fi
}


# set paths for download
PATH_HOME="$(pwd)"
PATH_TF="$PATH_HOME/tensorflow"
PATH_BAZEL="$PATH_HOME/bazel"
PATH_CUDNN="$PATH_HOME"
PATH_MODEL="$PATH_HOME/model"

# download cudnn
if [ ! -f "$PATH_CUDNN/include/cudnn.h" ]; then
    echo "==> Failed to find $PATH_CUDNN/include/cudnn.h"
    exit
fi

if [ $(find $PATH_CUDNN/lib64 -name "libcudnn.so*" | wc -l) -eq 0 ]; then
    echo "==> Failed to find $PATH_CUDNN/lib64/libcudnn.so.x.x.x"
    exit
fi


# download inception if not exist
if [ ! -f "$PATH_MODEL/tensorflow_inception_graph.pb" ]; then
  wget_and_check \
   "$PATH_MODEL/inception_dec_2015.zip" \
   "https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip" \
   "abb06016685bbf7e849b707b00be2754"
  unzip $PATH_MODEL/inception_dec_2015.zip -d $PATH_MODEL
  rm -rf inception_dec_2015.zip
fi


# build bazel if not exist
if [ ! -f "$PATH_BAZEL/output/bazel" ]; then
  cd $PATH_BAZEL
  ./compile.sh
  cd $PATH_HOME
fi


# build tensorflow if not exist
if [ ! -f "$PATH_HOME/lib64/libtensorflow.so" ]; then

  # set env variables for tensorflow
  export PYTHON_BIN_PATH=$(which python)
  export TF_NEED_GCP="0"
  export TF_NEED_CUDA="1"
  export GCC_HOST_COMPILER_PATH=$(which gcc)
  export TF_CUDA_VERSION="$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')"
  export CUDA_TOOLKIT_PATH="/usr/local/cuda"
  export TF_CUDNN_VERSION="$(basename $(find lib64 -name "libcudnn.so*") | sed 's/libcudnn\.so\.//g')"
  export CUDNN_INSTALL_PATH="$PATH_CUDNN"
  export TF_CUDA_COMPUTE_CAPABILITIES="3.5,5.2" # K-40 (3.5), TITAN-X (5.2), GTX-1080 (6.1)


  # apply patch (bazel cant handle symlink, 151e2e7bbd41d16c88e802d6678b422d0847c2b8)
  cd $PATH_TF
  git reset --hard HEAD
  git checkout third_party/gpus/crosstool/CROSSTOOL
echo '
diff --git a/third_party/gpus/crosstool/CROSSTOOL b/third_party/gpus/crosstool/CROSSTOOL
index 8db81a9..8c08ed3 100644
--- a/third_party/gpus/crosstool/CROSSTOOL
+++ b/third_party/gpus/crosstool/CROSSTOOL
@@ -60,6 +60,7 @@ toolchain {
   cxx_builtin_include_directory: "/usr/lib/gcc/"
   cxx_builtin_include_directory: "/usr/local/include"
   cxx_builtin_include_directory: "/usr/include"
+  cxx_builtin_include_directory: "/usr/local/cuda-CUDA_VERSION/include"
   tool_path { name: "gcov" path: "/usr/bin/gcov" }
 
   # C(++) compiles invoke the compiler (as that is the one knowing where
' | sed "s/CUDA_VERSION/$TF_CUDA_VERSION/g" | git apply
 

  # build tensorflow
  ./configure
  $PATH_BAZEL/output/bazel build -c opt --config=cuda //tensorflow:libtensorflow_cc.so
  if [ $? -ne 0 ]; then
    cd $PATH_HOME
    echo "==> Failed to build Tensorflow. You may try again"
    exit
  fi


  # copy library and headers. better solution is needed
  # (sync does not work well with symlinked structure)
  mkdir -p $PATH_HOME/lib64 $PATH_HOME/include
  cp bazel-bin/tensorflow/libtensorflow_cc.so $PATH_HOME/lib64/libtensorflow.so

  # copy headers for tensorflow
  rsync -am --include='*.h' -f 'hide,! */' bazel-tensorflow/tensorflow/ $PATH_HOME/include/tensorflow
  rsync -am --include='*.h' -f 'hide,! */' bazel-genfiles/tensorflow/ $PATH_HOME/include/tensorflow

  # copy headers for protobuf
  rsync -am --include='*.h' -f 'hide,! */' bazel-tensorflow/external/protobuf/src/ $PATH_HOME/include

  # copy headers for eigen3 (this library header stucture is weird)
  mkdir -p $PATH_HOME/include/third_party/eigen3/unsupported/Eigen
  cp -rf bazel-tensorflow/external/eigen_archive/unsupported/Eigen/* $PATH_HOME/include/third_party/eigen3/unsupported/Eigen
  mkdir -p $PATH_HOME/include/third_party/eigen3/Eigen
  cp -rf bazel-tensorflow/external/eigen_archive/Eigen/* $PATH_HOME/include/third_party/eigen3/Eigen
  cp -rn bazel-tensorflow/third_party/eigen3/unsupported/Eigen/* $PATH_HOME/include/third_party/eigen3/unsupported/Eigen
  ln -sf $PATH_HOME/include/third_party/eigen3/Eigen $PATH_HOME/include/Eigen
  cd $PATH_HOME
fi


# use existing tensorflow example with a patch
if [ ! -f "$PATH_HOME/app.cc" ]; then
  cd $PATH_TF
  git checkout $PATH_TF/tensorflow/examples/label_image/main.cc
echo '
diff --git a/tensorflow/examples/label_image/main.cc b/tensorflow/examples/label_image/main.cc
index 7faf1ce..f02f5db 100644
--- a/tensorflow/examples/label_image/main.cc
+++ b/tensorflow/examples/label_image/main.cc
@@ -231,13 +231,9 @@ int main(int argc, char* argv[]) {
   // They define where the graph and input data is located, and what kind of
   // input the model expects. If you train your own model, or use something
   // other than GoogLeNet youSQll need to update these.
-  string image = "tensorflow/examples/label_image/data/grace_hopper.jpg";
-  string graph =
-      "tensorflow/examples/label_image/data/"
-      "tensorflow_inception_graph.pb";
-  string labels =
-      "tensorflow/examples/label_image/data/"
-      "imagenet_comp_graph_label_strings.txt";
+  string image = "tensorflow/tensorflow/examples/label_image/data/grace_hopper.jpg";
+  string graph = "model/tensorflow_inception_graph.pb";
+  string labels = "model/imagenet_comp_graph_label_strings.txt";
   int32 input_width = 299;
   int32 input_height = 299;
   int32 input_mean = 128;
' | sed "s/SQ/\'/g" | git apply
  cp $PATH_TF/tensorflow/examples/label_image/main.cc $PATH_HOME/app.cc
  cd $PATH_HOME
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PATH_HOME/lib64
fi
echo "==> Done"
