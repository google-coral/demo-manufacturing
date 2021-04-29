// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

ClassificationResult InferenceWrapper::get_classification_result(
    const uint8_t* input_data, const int input_size) {
  std::vector<float> output_data;
  uint8_t* input = interpreter_->typed_input_tensor<uint8_t>(0);
  std::memcpy(input, input_data, input_size);

  CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);

  const auto& output_indices = interpreter_->outputs();
  const auto* out_tensor = interpreter_->tensor(output_indices[0]);

  float max_prob;
  int max_index;
  // Handles only uint8 or float outputs.
  if (out_tensor->type == kTfLiteUInt8) {
    const uint8_t* output = interpreter_->typed_output_tensor<uint8_t>(0);
    max_index = std::max_element(output, output + out_tensor->bytes) - output;
    // For uint8 output, we need to apply zero point amd scale.
    max_prob = (output[max_index] - out_tensor->params.zero_point) * out_tensor->params.scale;
  } else if (out_tensor->type == kTfLiteFloat32) {
    const float* output = interpreter_->typed_output_tensor<float>(0);
    max_index = std::max_element(output, output + out_tensor->bytes / sizeof(float)) - output;
    max_prob = output[max_index];
  } else {
    std::cerr << "Tensor " << out_tensor->name
              << " has unsupported output type: " << out_tensor->type << std::endl;
    exit(EXIT_FAILURE);
  }
  return {labels_[max_index], max_prob};
}

std::vector<DetectionResult> InferenceWrapper::get_detection_results(
    const uint8_t* input_data, const int input_size, const float threshold,
    const std::vector<int>& want_ids) {
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
    if (out_tensor->type == kTfLiteFloat32) {
      // detection model out is float32
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
  return parse_detection_outputs(output_data, threshold, want_ids);
}

std::vector<DetectionResult> InferenceWrapper::parse_detection_outputs(
    const std::vector<std::vector<float>>& raw_output, const float threshold,
    const std::vector<int>& want_ids) {
  std::vector<DetectionResult> results;
  int n = lround(raw_output[3][0]);
  for (int i = 0; i < n; i++) {
    if (int id = lround(raw_output[1][i]); std::count(want_ids.begin(), want_ids.end(), id)) {
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
  }
  return results;
}

}  // namespace coral
