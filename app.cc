#include <fstream>
#include <vector>
#include <string>

#include "class_name.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(string file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";
  auto file_reader = tensorflow::ops::ReadFile(root.WithOpName(input_name),
                                               file_name);
  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
    image_reader = DecodeGif(root.WithOpName("gif_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, out_tensors));
  return Status::OK();
}

int main(int argc, char* argv[]) {

  string graph_path = "./data/tensorflow_inception_graph.pb";
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  tensorflow::GraphDef graph_def;
  if (!ReadBinaryProto(tensorflow::Env::Default(), graph_path, &graph_def).ok()) {
    LOG(ERROR) << "Read proto";
    return -1;
  }

  std::unique_ptr<tensorflow::Session> session;
  tensorflow::SessionOptions sess_opt;
  sess_opt.config.mutable_gpu_options()->set_allow_growth(true);
  (&session)->reset(tensorflow::NewSession(sess_opt));
  if (!session->Create(graph_def).ok()) {
    LOG(ERROR) << "Create graph";
    return -1;
  }

  const int batch_size = argc - 1;
  if (batch_size != 1) {
    LOG(ERROR) << "Batch mode for the pretrained inception-v3 is unsupported";
    LOG(ERROR) << " - https://github.com/tensorflow/tensorflow/issues/554";
    return -1;
  }

  int32 input_dim = 299;
  int32 input_mean = 128;
  int32 input_std = 128;
  std::vector<Tensor> inputs;
  std::string image_path(argv[1]);
  if (!ReadTensorFromImageFile(image_path, input_dim, input_dim, input_mean,
                               input_std, &inputs).ok()) {
    LOG(ERROR) << "Load image";
    return -1;
  }

  std::vector<Tensor> outputs;
  string input_layer = "Mul";
  string output_layer = "softmax";
  if (!session->Run({{input_layer, inputs[0]}},
                     {output_layer}, {}, &outputs).ok()) {
    LOG(ERROR) << "Running model failed";
    return -1;
  }

  Eigen::Map<Eigen::VectorXf> pred(outputs[0].flat<float>().data(),
                                   outputs[0].NumElements());
  int maxIndex; float maxValue = pred.maxCoeff(&maxIndex);
  LOG(INFO) << "P( " << class_name[maxIndex] << " | image ) = " << maxValue;

  return 0;
}
