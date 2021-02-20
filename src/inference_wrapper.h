#ifndef MANUFACTURING_DEMO_INFERENCE_WRAPPER_H_
#define MANUFACTURING_DEMO_INFERENCE_WRAPPER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "image_utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tflite/public/edgetpu.h"

namespace coral {

// Represents a Detection Result.
struct DetectionResult {
  std::string candidate;
  float score, x1, y1, x2, y2;
};

// Represents a Classification Result.
struct ClassificationResult {
  std::string candidate;
  float score;
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

  // Runs inference using given `interpreter` and get classification results
  ClassificationResult get_classification_result(const uint8_t* input_data, const int input_size);
  // Runs inference using given `interpreter` and get detection results.
  // want_ids contains the ids of the object that we want to filter.
  // 0 == person
  // 52 == apple
  std::vector<DetectionResult> get_detection_results(
      const uint8_t* input_data, const int input_size, const float threshold,
      const std::vector<int>& want_ids = {0, 52});
  // Helper function to parse ssd outputs into detection objects.
  std::vector<DetectionResult> parse_detection_outputs(
      const std::vector<std::vector<float>>& raw_output, const float threshold,
      const std::vector<int>& want_ids);
  // Get the input size of the model.
  size_t get_input_size() { return input_size_; }
  // Get the interpreter
  std::unique_ptr<tflite::Interpreter>& get_interpreter() { return interpreter_; }

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
#endif  // MANUFACTURING_DEMO_INFERENCE_WRAPPER_H_
