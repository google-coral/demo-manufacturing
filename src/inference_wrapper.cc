#include "inference_wrapper.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#define CORAL_ASSERT(x, y)         \
  {                                \
    if (!(x)) {                    \
      std::cerr << y << std::endl; \
      std::terminate();            \
    }                              \
  }

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                      \
  }

namespace coral {

namespace {
std::vector<std::string> read_labels(const std::string& label_path) {
  std::vector<std::string> labels;

  std::ifstream label_file(label_path);
  std::string line;
  if (label_file.is_open()) {
    while (getline(label_file, line)) {
      labels.push_back(line);
    }
  } else {
    std::cerr << "Unable to open file " << label_path << std::endl;
    exit(EXIT_FAILURE);
  }

  return labels;
}

}  // namespace
InferenceWrapper::InferenceWrapper(const std::string& model_path,
                                   const std::string& label_path) {
  tpu_context_ = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  TFLITE_MINIMAL_CHECK(model_ != nullptr);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  CORAL_ASSERT(
      tflite::InterpreterBuilder(*model_, resolver)(&interpreter_) == kTfLiteOk,
      "Failed to build Interpreter");
  interpreter_->SetExternalContext(kTfLiteEdgeTpuContext, tpu_context_.get());
  interpreter_->SetNumThreads(1);
  CORAL_ASSERT(interpreter_->AllocateTensors() == kTfLiteOk,
               "AllocateTensors failed");
  // sets output tensor shape.
  const auto& out_tensor_indices = interpreter_->outputs();
  output_shape_.resize(out_tensor_indices.size());
  for (size_t i = 0; i < out_tensor_indices.size(); ++i) {
    const auto* tensor = interpreter_->tensor(out_tensor_indices[i]);
    // For detection the output tensors are only of type float.
    output_shape_[i] = tensor->bytes / sizeof(float);
  }
  // Gets input size from interpeter, assumes square.
  input_size_ = interpreter_->input_tensor(0)->dims->data[1];
  labels_ = read_labels(label_path);
}

std::vector<std::vector<float>> InferenceWrapper::RunInference(
    const uint8_t* input_data, int input_size) {
  std::vector<std::vector<float>> output_data;
  uint8_t* input = interpreter_->typed_input_tensor<uint8_t>(0);
  std::memcpy(input, input_data, input_size);
  TFLITE_MINIMAL_CHECK(interpreter_->Invoke() == kTfLiteOk);
  const auto& output_indices = interpreter_->outputs();
  const size_t num_outputs = output_indices.size();
  output_data.resize(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = interpreter_->tensor(output_indices[i]);
    CORAL_ASSERT(out_tensor != nullptr, "unexpected null out_tensor");
    if (out_tensor->type == kTfLiteFloat32) {  // detection model out is Float32
      const size_t num_values = out_tensor->bytes / sizeof(float);

      const float* output = interpreter_->typed_output_tensor<float>(i);
      const size_t size_of_output_tensor_i = output_shape_[i];

      output_data[i].resize(size_of_output_tensor_i);
      for (size_t j = 0; j < size_of_output_tensor_i; ++j) {
        output_data[i][j] = output[j];
      }
    } else {
      std::cerr << "Unsupported output type: " << out_tensor->type
                << "\n Tensor Name: " << out_tensor->name;
    }
  }
  return output_data;
}

}  // namespace coral
