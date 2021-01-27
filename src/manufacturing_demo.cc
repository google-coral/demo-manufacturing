#include <sys/stat.h>

#include <cmath>
#include <fstream>
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
#include "keepout_shape.h"

using coral::Box;
using coral::CameraStreamer;
using coral::InferenceWrapper;
using coral::Point;
using coral::Polygon;

ABSL_FLAG(
    std::string, model, "models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite",
    "Path to tflite model.");
ABSL_FLAG(std::string, labels, "models/coco_labels.txt", "Path to labels file.");
ABSL_FLAG(std::string, input_path, "/dev/video0", "Path to video source or file to run inference.");
ABSL_FLAG(uint16_t, width, 640, "Input width.");
ABSL_FLAG(uint16_t, height, 480, "Input height.");
ABSL_FLAG(float, threshold, 0.5, "Minimum detection probability required to show bounding box.");
ABSL_FLAG(
    std::string, keepout_points_path, "",
    "If provided, detection boxes will be colored based on if they are "
    "in the keepout region (red) or not (green).");

namespace {

constexpr char* kSvgHeader = "<svg>";
constexpr char* kSvgFooter = "</svg>";
constexpr char* kSvgBox =
    "<rect x=\"$0\" y=\"$1\" width=\"$2\" height=\"$3\" "
    "fill-opacity=\"0.0\" "
    "style=\"stroke-width:5;stroke:rgb($4,$5,$6);\"/>";
constexpr char* kSvgText = "<text x=\"$0\" y=\"$1\" font-size=\"large\" fill=\"$2\">$3</text>";

// GStreamer definitions
#define LEAKY_Q " queue max-size-buffers=1 leaky=downstream "

void check_file(const char* file) {
  struct stat buf;
  if (stat(file, &buf) != 0) {
    LOG(ERROR) << file << " does not exist";
    exit(EXIT_FAILURE);
  }
}

// Callback function called from the appsink on new frames
void interpret_frame(
    const uint8_t* pixels, int pixel_length, GstElement* rsvg, InferenceWrapper& inferencer,
    int width, int height, float threshold, Polygon& keepout_polygon) {
  static int numFrames = 0;
  std::string boxlist;
  std::string labellist;
  const auto results = inferencer.GetDetectionResults(pixels, pixel_length, threshold);
  numFrames++;  // count number of frames processed
  VLOG(2) << "Frame: " << numFrames << " Candidates: " << results.size();

  std::string svg;
  for (const auto& result : results) {
    VLOG(3) << " - score: " << result.score << " x1: " << result.x1 * width
            << " y1: " << result.y1 * height << " x2: " << result.x2 * width
            << " y2: " << result.y2 * height << "\n";
    std::string box_str;
    std::string label_str;
    int w, h;
    w = (result.x2 - result.x1) * width;
    h = (result.y2 - result.y1) * height;
    // Checks if this box collided with the keepout.
    const auto& polygon_svg = keepout_polygon.get_svg_str();
    if (polygon_svg != "None") {
      // Check for keepout.
      Box b{result.x1 * width, result.y1 * height, result.x2 * width, result.y2 * height};
      if (b.collided_with_polygon(keepout_polygon, width)) {
        box_str = absl::Substitute(
            kSvgBox, result.x1 * width, result.y1 * height, w, h, 255, 0,
            0);  // Red
        label_str = absl::Substitute(
            kSvgText, result.x1 * width, (result.y1 * height) - 5, "red",
            absl::StrCat(result.candidate, ": ", result.score));
      } else {
        box_str = absl::Substitute(
            kSvgBox, result.x1 * width, result.y1 * height, w, h, 0, 255,
            0);  // Green
        label_str = absl::Substitute(
            kSvgText, result.x1 * width, (result.y1 * height) - 5, "lightgreen",
            absl::StrCat(result.candidate, ": ", result.score));
      }
    } else {
      // Don't check for keepout.
      box_str = absl::Substitute(
          kSvgBox, result.x1 * width, result.y1 * height, w, h, 0, 255,
          0);  // Green
      label_str = absl::Substitute(
          kSvgText, result.x1 * width, (result.y1 * height) - 5, "lightgreen",
          absl::StrCat(result.candidate, ": ", result.score));
    }
    boxlist = absl::StrCat(boxlist, box_str);
    labellist = absl::StrCat(labellist, label_str);
  }
  if (keepout_polygon.get_svg_str() == "None") {
    svg = absl::StrCat(kSvgHeader, boxlist, labellist, kSvgFooter);
  } else {
    svg = absl::StrCat(kSvgHeader, keepout_polygon.get_svg_str(), boxlist, labellist, kSvgFooter);
  }
  VLOG(3) << svg;
  // Push SVG data into the gstreamer module rsvgoverlay.
  g_object_set(G_OBJECT(rsvg), "data", svg.c_str(), NULL);
}

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  VLOG(1) << "Starting Manufacturing Demo\n";
  absl::ParseCommandLine(argc, argv);

  std::string model_path = absl::GetFlag(FLAGS_model);
  std::string label_path = absl::GetFlag(FLAGS_labels);
  const uint16_t width = absl::GetFlag(FLAGS_width);
  const uint16_t height = absl::GetFlag(FLAGS_height);
  const float threshold = absl::GetFlag(FLAGS_threshold);

  check_file(model_path.c_str());
  check_file(label_path.c_str());

  InferenceWrapper inferencer(model_path, label_path);
  coral::CameraStreamer streamer;
  size_t input_size = inferencer.GetInputSize();
  const auto input_path = absl::GetFlag(FLAGS_input_path);

  std::string pipeline;
  if (absl::StrContains(input_path, "/dev/video")) {
    pipeline = absl::StrFormat(
        "v4l2src device=%s !"
        "video/x-raw,framerate=30/1,width=%d,height=%d ! " LEAKY_Q " ! tee name=t t. !" LEAKY_Q
        "! videoconvert ! rsvgoverlay name=rsvg ! videoconvert ! "
        "glimagesink t. !" LEAKY_Q
        "! videoscale ! video/x-raw,width=%d,height=%d ! videoconvert ! "
        "video/x-raw,format=RGB ! appsink name=appsink",
        input_path, width, height, input_size, input_size);
  } else {
    // Assuming that input is a video.
    pipeline = absl::StrFormat(
        "filesrc location=%s ! decodebin ! tee name=t t. ! queue ! videoconvert ! "
        "videoscale ! video/x-raw,width=%d,height=%d ! rsvgoverlay name=rsvg ! "
        "videoconvert ! autovideosink t. ! videoconvert ! videoscale ! "
        "video/x-raw,width=%d,height=%d,format=RGB ! appsink name=appsink",
        input_path, width, height, input_size, input_size);
  }

  const gchar* kPipeline = pipeline.c_str();

  VLOG(2) << "Pipeline: " << pipeline.c_str();

  auto keepout_polygon = coral::parse_keepout_polygon(absl::GetFlag(FLAGS_keepout_points_path));
  streamer.RunPipeline(
      kPipeline, {interpret_frame, nullptr, inferencer, width, height, threshold, keepout_polygon});
}
