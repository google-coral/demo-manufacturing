#ifndef EDGETPU_CPP_EXAMPLES_UTILS_H_
#define EDGETPU_CPP_EXAMPLES_UTILS_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tflite/public/edgetpu.h"

namespace coral {

// Represents a Detection Result.
struct DetectionResult {
  std::string candidate;
  float score, x1, y1, x2, y2;
};

// A tflite::Interpreter wrapper class with extra features to parses
// Dectection models with ssd head.
class InferenceWrapper {
public:
  ~InferenceWrapper() = default;
  // Constructor for InferenceWrapper.
  InferenceWrapper(const std::string& model_path, const std::string& label_path);
  // InferenceWrapper is neither copyable nor movable.
  InferenceWrapper(const InferenceWrapper&) = delete;
  InferenceWrapper& operator=(const InferenceWrapper&) = delete;

  // Runs inference using given `interpreter`
  std::vector<DetectionResult> GetDetectionResults(
      const uint8_t* input_data, const int input_size, const float threshold);
  // Helper function to parse ssd outputs into detection objects.
  std::vector<DetectionResult> ParseOutputs(
      const std::vector<std::vector<float>>& raw_output, const float threshold);
  // Get the input size of the model.
  size_t GetInputSize() { return input_size_; };

private:
  InferenceWrapper() = default;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::map<int, std::string> labels_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  std::shared_ptr<edgetpu::EdgeTpuContext> tpu_context_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  size_t input_size_;
};

}  // namespace coral
#endif  // EDGETPU_CPP_EXAMPLES_UTILS_H_
