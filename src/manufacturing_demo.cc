#include <sys/stat.h>

#include <iostream>
#include <memory>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"
#include "camera_streamer.h"
#include "glog/logging.h"
#include "inference_wrapper.h"

using coral::CameraStreamer;
using coral::InferenceWrapper;

ABSL_FLAG(std::string, model, "default.tflite", "Path to tflite model.");
ABSL_FLAG(std::string, labels, "default.txt", "Path to labels file");
ABSL_FLAG(uint16_t, width, 640, "Input width.");
ABSL_FLAG(uint16_t, height, 480, "Input height.");

// GStreamer definitions
#define LEAKY_Q " queue max-size-buffers=1 leaky=downstream "

// Callback function called from the appsink on new frames
void interpret_frame(const uint8_t* pixels, int length, void* args) {
  InferenceWrapper* inferencer = reinterpret_cast<InferenceWrapper*>(args);

  inferencer->RunInference(pixels, length);
}

void check_file(const char* file) {
  struct stat buf;
  if (stat(file, &buf) != 0) {
    LOG(ERROR) << file << " does not exist";
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  std::string model_path = absl::GetFlag(FLAGS_model);
  std::string label_path = absl::GetFlag(FLAGS_labels);
  uint16_t width = absl::GetFlag(FLAGS_width);
  uint16_t height = absl::GetFlag(FLAGS_height);

  check_file(model_path.c_str());
  check_file(label_path.c_str());

  InferenceWrapper inferencer(model_path, label_path);
  coral::CameraStreamer streamer;
  size_t input_size = inferencer.GetInputSize();
  const std::string pipeline = absl::StrFormat(
      "v4l2src device = /dev/video0 !"
      "video/x-raw,framerate=30/1,width=%d,height=%d ! " LEAKY_Q
      " ! tee name=t"
      " t. !" LEAKY_Q
      "! glimagesink"
      " t. !" LEAKY_Q
      "! videoscale ! video/x-raw,width=%d,height=%d ! videoconvert ! "
      "video/x-raw,format=RGB ! appsink name=appsink",
      width, height, input_size, input_size);
  const gchar* kPipeline = pipeline.c_str();

  streamer.RunPipeline(kPipeline, {interpret_frame, &inferencer});
}
