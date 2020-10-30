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

ABSL_FLAG(std::string, model,
          "mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite",
          "Path to tflite model.");
ABSL_FLAG(std::string, labels, "coco_labels.txt", "Path to labels file.");
ABSL_FLAG(uint16_t, width, 640, "Input width.");
ABSL_FLAG(uint16_t, height, 480, "Input height.");
ABSL_FLAG(float, threshold, 0.5,
          "Minimum detection probability required to show bounding box.");
ABSL_FLAG(std::string, keepout_points_path, "",
          "If provided, detection boxes will be colored based on if they are "
          "in the keepout region (red) or not (green).");

constexpr char *kSvgHeader = "<svg>";
constexpr char *kSvgFooter = "</svg>";
constexpr char *kSvgBox =
    "<rect x=\"$0\" y=\"$1\" width=\"$2\" height=\"$3\" fill-opacity=\"0.0\" "
    "style=\"stroke-width:5;stroke:rgb($4,$5,$6);\"/>";

// GStreamer definitions
#define LEAKY_Q " queue max-size-buffers=1 leaky=downstream "

// Callback function called from the appsink on new frames
void interpret_frame(const uint8_t *pixels, int length, GstElement *rsvg,
                     void *args, int width, int height, float threshold,
                     Polygon keepout_polygon) {
  static int numFrames = 0;
  std::string boxlist;
  InferenceWrapper *inferencer = reinterpret_cast<InferenceWrapper *>(args);
  std::vector<std::vector<float>> results =
      inferencer->RunInference(pixels, length);
  numFrames++; // count number of frames processed
  VLOG(2) << "frame: " << numFrames;
  int ndet = (int)results[3][0]; // number of detected objects in the frame
  int x1, y1, x2, y2, w, h;

  VLOG(3) << "#####frame " << numFrames
          << "  number detected: " << results[3][0]
          << " screen width: " << width;

  if (ndet > 0) {
    std::string svg;
    for (int i = 0; i < ndet; i++) {
      VLOG(3) << "index " << i << " class:" << results[1][i]
              << " score:" << results[2][i] << "  top:" << results[0][i * 4]
              << "  left:" << results[0][i * 4 + 1]
              << " bottom:" << results[0][i * 4 + 2]
              << "  right:" << results[0][i * 4 + 3]
              << "   threshold: " << threshold;
      if (results[2][i] >= threshold) {
        y1 = std::max(static_cast<int>(std::round(height * results[0][i * 4])),
                      0); // Top should not be less than 0.
        x1 = std::max(
            static_cast<int>(std::round(width * results[0][i * 4 + 1])),
            0); // Left should not be less than 0.
        y2 = std::min(
            static_cast<int>(std::round(height * results[0][i * 4 + 2])),
            height); // Bottom should not be higher than height.
        x2 = std::min(
            static_cast<int>(std::round(width * results[0][i * 4 + 3])),
            width); // Right should not be higher than width.
        w = x2 - x1;
        h = y2 - y1;
        VLOG(4) << "x1,y1,w,h " << x1 << " " << y1 << " " << w << " " << h;
      }
      std::string box0;
      // Checks if this box collided with the keepout.
      Box b{x1, y1, x2, y2};
      const auto &polygon_svg = keepout_polygon.get_svg_str();
      if (polygon_svg != "None") {
        if (b.collided_with_polygon(keepout_polygon, width)) {
          box0 = absl::Substitute(kSvgBox, x1, y1, w, h, 255, 0, 0); // Red
        } else {
          box0 = absl::Substitute(kSvgBox, x1, y1, w, h, 0, 255, 0); // Green
        }
        svg = absl::StrCat(kSvgHeader, polygon_svg, boxlist, kSvgFooter);
      } else {
        box0 = absl::Substitute(kSvgBox, x1, y1, w, h, 0, 255, 0); // Green
        svg = absl::StrCat(kSvgHeader, boxlist, kSvgFooter);
      }
      boxlist = absl::StrCat(boxlist, box0);
      VLOG(5) << "boxlist: " << boxlist;
    }
    VLOG(6) << "svg: " << svg;
    // Push SVG data into the gstreamer module rsvgoverlay.
    g_object_set(G_OBJECT(rsvg), "data", svg.c_str(), NULL);
  }
}

void check_file(const char *file) {
  struct stat buf;
  if (stat(file, &buf) != 0) {
    LOG(ERROR) << file << " does not exist";
    exit(EXIT_FAILURE);
  }
}

Polygon parse_keepout_polygon(const std::string &file_path) {
  std::ifstream f{file_path};
  std::vector<Point> points;
  if (f.is_open()) {
    std::string polygon_svg;
    polygon_svg = "<polygon points=\"";
    // Ignores csv header.
    f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    for (std::string line; std::getline(f, line);) {
      polygon_svg = absl::StrCat(polygon_svg, " ", line);
      int x, y;
      std::vector<std::string> p = absl::StrSplit(line, ',');
      absl::SimpleAtoi(p[0], &x);
      absl::SimpleAtoi(p[1], &y);
      points.emplace_back(x, y);
    }
    polygon_svg = absl::StrCat(
        polygon_svg, " \" style=\"fill:none;stroke:green;stroke-width:5\" /> ");
    Polygon keepout_polygon(points);
    keepout_polygon.set_svg_str(polygon_svg);
    return keepout_polygon;
  }
  return {};
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "starting manufacturing_demo ";

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
  const std::string pipeline = absl::StrFormat(
      "v4l2src device = /dev/video0 !" // TODO(riceg) support filesrc also
      "video/x-raw,framerate=30/1,width=%d,height=%d ! " LEAKY_Q " ! tee name=t"
      " t. !" LEAKY_Q
      "! videoconvert ! rsvgoverlay name=rsvg ! videoconvert ! glimagesink "
      " t. !" LEAKY_Q
      "! videoscale ! video/x-raw,width=%d,height=%d ! videoconvert ! "
      "video/x-raw,format=RGB ! appsink name=appsink",
      width, height, input_size, input_size);
  const gchar *kPipeline = pipeline.c_str();

  LOG(INFO) << "gstreamer pipeline: " << pipeline.c_str();
  LOG(INFO) << "size: " << input_size;
  LOG(INFO) << "Threshold: " << threshold;

  auto keepout_polygon =
      parse_keepout_polygon(absl::GetFlag(FLAGS_keepout_points_path));
  streamer.RunPipeline(kPipeline, {interpret_frame, nullptr, &inferencer, width,
                                   height, threshold, keepout_polygon});
}
