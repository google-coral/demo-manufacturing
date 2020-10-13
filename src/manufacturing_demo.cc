#include <sys/stat.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "camera_streamer.h"
#include "glog/logging.h"
#include "inference_wrapper.h"

using coral::CameraStreamer;
using coral::InferenceWrapper;

ABSL_FLAG(std::string, model,
          "mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite",
          "Path to tflite model.");
ABSL_FLAG(std::string, labels, "coco_labels.txt", "Path to labels file");
ABSL_FLAG(uint16_t, width, 640, "Input width.");
ABSL_FLAG(uint16_t, height, 480, "Input height.");
ABSL_FLAG(float, threshold, 0.5,
          "Minimum detection probability required to show bounding box");

const char* kSvgHeader = "<svg>";
const char* kSvgFooter = "</svg>";
const char* kSvgBox =
    "<rect x=\"$0\" y=\"$1\" width=\"$2\" height=\"$3\" fill-opacity=\"0.1\" "
    "style=\"stroke-width:5;stroke:rgb($4,$5,$6);\"/>";
// GStreamer definitions
#define LEAKY_Q " queue max-size-buffers=1 leaky=downstream "

int numFrames = 0;
// Callback function called from the appsink on new frames
void interpret_frame(const uint8_t* pixels, int length, GstElement* rsvg,
                     void* args) {
  static int width = absl::GetFlag(FLAGS_width);
  static int height = absl::GetFlag(FLAGS_height);
  static float threshold = absl::GetFlag(FLAGS_threshold);
  std::string boxlist;
  InferenceWrapper* inferencer = reinterpret_cast<InferenceWrapper*>(args);
  std::vector<std::vector<float>> results =
      inferencer->RunInference(pixels, length);
  numFrames++;  // count number of frames processed
  VLOG(2) << "frame: " << numFrames;
  int ndet = (int)results[3][0];  // number of detected objects in the frame
  int x = 50, y = 50, w = 5, h = 5;

  VLOG(3) << "#####frame " << numFrames
          << "  number detected: " << results[3][0]
          << " screen width: " << width;

  if (ndet > 0) {
    for (int i = 0; i < ndet; i++) {
      VLOG(3) << "index " << i << " class:" << results[1][i]
              << " score:" << results[2][i] << "  top:" << results[0][i * 4]
              << "  left:" << results[0][i * 4 + 1]
              << " bottom:" << results[0][i * 4 + 2]
              << "  right:" << results[0][i * 4 + 3]
              << "   threshold: " << threshold;
      if (results[2][i] >= threshold) {
        y = static_cast<int>(std::round(width * results[0][i * 4]));  // top
        x = static_cast<int>(
            std::round(height * results[0][i * 4 + 1]));  // left
        h = static_cast<int>(std::round(width * results[0][i * 4 + 2])) -
            y;  // bottom-top
        w = static_cast<int>(std::round(height * results[0][i * 4 + 3])) -
            x;  // right-left
        VLOG(4) << "x,y,w,h= " << x << " " << y << " " << w << " " << h;
      }
      std::string box0;
      box0 = absl::Substitute(kSvgBox, x, y, w, h, 0, 255, 0);  // Green
      boxlist = absl::StrCat(boxlist, box0);
      VLOG(5) << "boxlist: " << boxlist;
    }
    std::string svg = absl::StrCat(kSvgHeader, boxlist, kSvgFooter);
    VLOG(6) << "svg: " << svg;
    // Push SVG data into the gstreamer module rsvgoverlay.
    g_object_set(G_OBJECT(rsvg), "data", svg.c_str(), NULL);
  }
}

void check_file(const char* file) {
  struct stat buf;
  if (stat(file, &buf) != 0) {
    LOG(ERROR) << file << " does not exist";
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "starting manufacturing_demo ";

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
      "v4l2src device = /dev/video0 !" //TODO(riceg) support filesrc also
      "video/x-raw,framerate=30/1,width=%d,height=%d ! " LEAKY_Q
      " ! tee name=t"
      " t. !" LEAKY_Q
      "! videoconvert ! rsvgoverlay name=rsvg ! videoconvert ! glimagesink "
      " t. !" LEAKY_Q
      "! videoscale ! video/x-raw,width=%d,height=%d ! videoconvert ! "
      "video/x-raw,format=RGB ! appsink name=appsink",
      width, height, input_size, input_size);
  const gchar* kPipeline = pipeline.c_str();

  LOG(INFO) << "gstreamer pipeline: " << pipeline.c_str();
  LOG(INFO) << "size: " << input_size;
  LOG(INFO) << "Threshold: " << absl::GetFlag(FLAGS_threshold);

  streamer.RunPipeline(kPipeline, {interpret_frame, nullptr, &inferencer});
}
