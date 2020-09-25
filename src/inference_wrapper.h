#ifndef EDGETPU_CPP_EXAMPLES_UTILS_H_
#define EDGETPU_CPP_EXAMPLES_UTILS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tflite/public/edgetpu.h"

namespace coral {

class InferenceWrapper {
 public:
  ~InferenceWrapper() = default;

  InferenceWrapper(const std::string& model_path,
                   const std::string& label_path);

  // InferenceWrapper is neither copyable nor movable
  InferenceWrapper(const InferenceWrapper&) = delete;
  InferenceWrapper& operator=(const InferenceWrapper&) = delete;

  // Runs inference using given `interpreter`
  std::vector<std::vector<float>> RunInference(const uint8_t* input_data,
                                               int input_size);
  size_t GetInputSize() { return input_size_; };

 private:
  InferenceWrapper() = default;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::vector<std::string> labels_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  std::shared_ptr<edgetpu::EdgeTpuContext> tpu_context_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  size_t input_size_;
};

}  // namespace coral
#endif  // EDGETPU_CPP_EXAMPLES_UTILS_H_
