#include "inference_wrapper.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <string>

#include "glog/logging.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace coral {

namespace {

void read_labels(std::map<int, std::string>& labels, const std::string& label_path) {
  std::ifstream label_file(label_path);
  if (!label_file.good()) {
    LOG(ERROR) << "Unable to open file " << label_path;
    exit(EXIT_FAILURE);
  }
  for (std::string line; getline(label_file, line);) {
    std::istringstream ss(line);
    int id;
    ss >> id;
    // Trim the id and the space from the line to get label.
    line = std::regex_replace(line, std::regex("^[0-9]+ +"), "");
    labels.emplace(id, line);
  }
}
}  // namespace
InferenceWrapper::InferenceWrapper(const std::string& model_path, const std::string& label_path) {
  tpu_context_ = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  CHECK_NOTNULL(model_);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  CHECK_EQ(tflite::InterpreterBuilder(*model_, resolver)(&interpreter_), kTfLiteOk)
      << "Failed to build Interpreter";
  interpreter_->SetExternalContext(kTfLiteEdgeTpuContext, tpu_context_.get());
  interpreter_->SetNumThreads(1);
  CHECK_EQ(interpreter_->AllocateTensors(), kTfLiteOk) << "AllocateTensors failed";

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
  read_labels(labels_, label_path);
}

std::vector<DetectionResult> InferenceWrapper::GetDetectionResults(
    const uint8_t* input_data, const int input_size, const float threshold) {
  std::vector<std::vector<float>> output_data;

  uint8_t* input = interpreter_->typed_input_tensor<uint8_t>(0);
  std::memcpy(input, input_data, input_size);

  CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);

  const auto& output_indices = interpreter_->outputs();
  const size_t num_outputs = output_indices.size();
  output_data.resize(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = interpreter_->tensor(output_indices[i]);
    CHECK_NOTNULL(out_tensor);
    if (out_tensor->type == kTfLiteFloat32) {  // detection model out is Float32
      const size_t num_values = out_tensor->bytes / sizeof(float);

      const float* output = interpreter_->typed_output_tensor<float>(i);
      const size_t size_of_output_tensor_i = output_shape_[i];

      output_data[i].resize(size_of_output_tensor_i);
      for (size_t j = 0; j < size_of_output_tensor_i; ++j) {
        output_data[i][j] = output[j];
      }
    } else {
      LOG(ERROR) << "Unsupported output type: " << out_tensor->type
                 << "\n Tensor Name: " << out_tensor->name;
    }
  }
  return ParseOutputs(output_data, threshold);
}

std::vector<DetectionResult> InferenceWrapper::ParseOutputs(
    const std::vector<std::vector<float>>& raw_output, const float threshold) {
  std::vector<DetectionResult> results;
  int n = lround(raw_output[3][0]);
  for (int i = 0; i < n; i++) {
    int id = lround(raw_output[1][i]);
    float score = raw_output[2][i];
    if (score > threshold) {
      DetectionResult result;
      result.candidate = labels_.at(id);
      result.score = score;
      result.y1 = std::max(static_cast<float>(0.0), raw_output[0][4 * i]);
      result.x1 = std::max(static_cast<float>(0.0), raw_output[0][4 * i + 1]);
      result.y2 = std::min(static_cast<float>(1.0), raw_output[0][4 * i + 2]);
      result.x2 = std::min(static_cast<float>(1.0), raw_output[0][4 * i + 3]);
      results.push_back(result);
    }
  }
  return results;
}

}  // namespace coral
