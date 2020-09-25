#include <sys/stat.h>

#include <iostream>
#include <memory>
#include <vector>

#include "absl/strings/str_format.h"
#include "camera_streamer.h"
#include "inference_wrapper.h"

using coral::CameraStreamer;
using coral::InferenceWrapper;

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
    std::cerr << file << " does not exist" << std::endl;
    exit(EXIT_FAILURE);
  }
}

void usage(char* argv[]) {
  std::cerr << "Usage: " << argv[0] << " --model model_file --labels label_file"
            << std::endl;
  exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
  std::string model_path;
  std::string label_path;

  if (argc == 5) {
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--model")
        model_path = argv[++i];
      else if (std::string(argv[i]) == "--labels")
        label_path = argv[++i];
      else
        usage(argv);
    }
  } else {
    usage(argv);
  }

  check_file(model_path.c_str());
  check_file(label_path.c_str());

  InferenceWrapper inferencer(model_path, label_path);
  coral::CameraStreamer streamer;
  size_t input_size = inferencer.GetInputSize();
  const std::string pipeline = absl::StrFormat(
      "v4l2src device = /dev/video0 !"
      "video/x-raw,framerate=30/1,width=640,height=480 ! " LEAKY_Q
      " ! tee name=t"
      " t. !" LEAKY_Q
      "! glimagesink"
      " t. !" LEAKY_Q
      "! videoscale ! video/x-raw,width=%zu,height=%zu ! videoconvert ! "
      "video/x-raw,format=RGB ! appsink name=appsink",
      input_size, input_size);
  const gchar* kPipeline = pipeline.c_str();

  streamer.RunPipeline(kPipeline, {interpret_frame, &inferencer});
}
