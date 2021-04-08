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
#include "image_utils.h"
#include "inference_wrapper.h"
#include "keepout_shape.h"

using coral::Box;
using coral::CameraStreamer;
using coral::InferenceWrapper;
using coral::Point;
using coral::Polygon;

ABSL_FLAG(
    std::string, detection_model, "models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite",
    "Path to detection model.");
ABSL_FLAG(
    std::string, detection_labels, "models/coco_labels.txt", "Path to detection labels file.");
ABSL_FLAG(
    std::string, classifier_model, "models/classifier_edgetpu.tflite",
    "Path to classification model.");
ABSL_FLAG(
    std::string, classifier_labels, "models/classifier_labels.txt",
    "Path to classification labels file.");
ABSL_FLAG(
    std::string, worker_safety_input, "test_data/worker-zone-detection.mp4",
    "Path to video source or file to run worker safety inference.");
ABSL_FLAG(
    std::string, visual_inspection_input, "test_data/apple.mp4",
    "Path to video source or file to run visual inspection inference.");
ABSL_FLAG(uint16_t, width, 960, "Width to scale both inputs to.");
ABSL_FLAG(uint16_t, height, 540, "Height to scale both inputs to.");
ABSL_FLAG(float, threshold, 0.5, "Minimum detection probability required to show bounding box.");
ABSL_FLAG(
    std::string, keepout_points_path, "config/keepout_points.csv",
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
}  // namespace

namespace callback_helper {
// Callback function for the manufacturing demo called from the appsink on every new frame
void worker_safety_callback(
    GstElement* rsvg, const uint8_t* pixels, int pixel_length, InferenceWrapper& detector,
    int width, int height, float threshold, Polygon& keepout_polygon) {
  static int frame_num = 0;
  std::string box_list;
  std::string label_list;
  const auto results =
      detector.get_detection_results(pixels, pixel_length, threshold, /*want_ids=*/{0});
  frame_num++;  // count number of frames processed
  VLOG(4) << "Frame: " << frame_num << " Candidates: " << results.size();

  std::string svg;
  for (const auto& result : results) {
    VLOG(5) << " - score: " << result.score << " x1: " << result.x1 * width
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
    box_list = absl::StrCat(box_list, box_str);
    label_list = absl::StrCat(label_list, label_str);
  }
  if (keepout_polygon.get_svg_str() == "None") {
    svg = absl::StrCat(kSvgHeader, box_list, label_list, kSvgFooter);
  } else {
    svg = absl::StrCat(kSvgHeader, keepout_polygon.get_svg_str(), box_list, label_list, kSvgFooter);
  }
  VLOG(5) << svg;
  // Push SVG data into the gstreamer module rsvgoverlay.
  g_object_set(G_OBJECT(rsvg), "data", svg.c_str(), NULL);
}

// Callback function for the visual inspection demo called from the appsink on every new frame
void visual_inspection_callback(
    GstElement* rsvg, uint8_t* pixels, int pixel_length, InferenceWrapper& detector,
    InferenceWrapper& classifier, int width, int height, float threshold) {
  static int frame_num = 0;
  std::string box_list;
  std::string label_list;
  const auto results =
      detector.get_detection_results(pixels, pixel_length, threshold, /*want_id*/ {52});
  frame_num++;  // count number of frames processed
  VLOG(4) << "Frame: " << frame_num << " Candidates: " << results.size();

  std::string svg;
  for (const auto& result : results) {
    VLOG(5) << " x1: " << result.x1 * width << " y1: " << result.y1 * height
            << " x2: " << result.x2 * width << " y2: " << result.y2 * height << "\n";
    std::string box_str;
    std::string label_str;
    int w = (result.x2 - result.x1) * width;
    int h = (result.y2 - result.y1) * height;

    const auto detector_input_size = detector.get_input_size();
    const coral::ImageDims image_dim{detector_input_size, detector_input_size, 3};
    const coral::BoundingBox crop_area{
        result.y1 * image_dim[0], result.x1 * image_dim[0], result.y2 * image_dim[1],
        result.x2 * image_dim[1]};
    const auto& cropped_image = coral::crop_image(pixels, image_dim, crop_area);

    const coral::ImageDims in_dim{crop_area.height, crop_area.width, 3};
    const coral::ImageDims out_dim{classifier.get_input_size(), classifier.get_input_size(), 3};
    const auto& resized_image = coral::resize_image(cropped_image.data(), in_dim, out_dim);

    const auto classification =
        classifier.get_classification_result(resized_image.data(), resized_image.size());
    if (classification.score > threshold) {
      VLOG(4) << classification.candidate << ": " << classification.score;
      if (classification.candidate == "fresh_apple") {
        // Fresh Apple.
        box_str = absl::Substitute(
            kSvgBox, result.x1 * width, result.y1 * height, w, h, 0, 255,
            0);  // Green
        label_str = absl::Substitute(
            kSvgText, result.x1 * width, (result.y1 * height) - 5, "lightgreen",
            absl::StrCat(classification.candidate, ": ", classification.score));
      } else {
        // Rotten Apple.
        box_str = absl::Substitute(
            kSvgBox, result.x1 * width, result.y1 * height, w, h, 255, 0,
            0);  // Red
        label_str = absl::Substitute(
            kSvgText, result.x1 * width, (result.y1 * height) - 5, "red",
            absl::StrCat(classification.candidate, ": ", classification.score));
      }
    }
    box_list = absl::StrCat(box_list, box_str);
    label_list = absl::StrCat(label_list, label_str);
  }
  svg = absl::StrCat(kSvgHeader, box_list, label_list, kSvgFooter);
  VLOG(5) << svg;
  // Push SVG data into the gstreamer module rsvgoverlay.
  g_object_set(G_OBJECT(rsvg), "data", svg.c_str(), NULL);
}

}  // namespace callback_helper

static std::string generate_pipeline_string(
    const std::string input_path, const uint16_t width, const uint16_t height,
    const size_t detector_input_size, const std::string demo_name) {
  std::string pipeline;
  if (absl::StrContains(input_path, "/dev/video")) {
    pipeline = absl::StrFormat(
        "v4l2src device=%s !"
        "video/x-raw,framerate=30/1,width=%d,height=%d ! " LEAKY_Q
        " ! tee name=t_%s "
        "t_%s. !" LEAKY_Q
        "! videoconvert ! rsvgoverlay name=rsvg_%s ! videoconvert ! m. \n"
        "t_%s. !" LEAKY_Q
        "! videoscale ! video/x-raw,width=%d,height=%d ! "
        "videoconvert ! video/x-raw,format=RGB ! appsink name=appsink_%s\n",
        input_path, width, height, demo_name, demo_name, demo_name, demo_name, detector_input_size,
        detector_input_size, demo_name);
  } else {
    // Assuming that input is a video.
    pipeline = absl::StrFormat(
        "filesrc location=%s ! decodebin ! tee name=t_%s "
        "t_%s. ! queue ! videoconvert ! videoscale ! video/x-raw,width=%d,height=%d ! "
        "rsvgoverlay name=rsvg_%s ! videoconvert ! m.\n"
        "t_%s. ! queue ! videoconvert ! videoscale ! "
        "video/x-raw,width=%d,height=%d,format=RGB ! appsink name=appsink_%s\n",
        input_path, demo_name, demo_name, width, height, demo_name, demo_name, detector_input_size,
        detector_input_size, demo_name);
  }
  return pipeline;
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  std::string detection_model_path = absl::GetFlag(FLAGS_detection_model);
  std::string detection_label_path = absl::GetFlag(FLAGS_detection_labels);
  std::string classifier_model_path = absl::GetFlag(FLAGS_classifier_model);
  std::string classifier_label_path = absl::GetFlag(FLAGS_classifier_labels);
  const uint16_t width = absl::GetFlag(FLAGS_width);
  const uint16_t height = absl::GetFlag(FLAGS_height);
  const float threshold = absl::GetFlag(FLAGS_threshold);

  check_file(detection_model_path.c_str());
  check_file(detection_label_path.c_str());
  check_file(classifier_label_path.c_str());
  check_file(classifier_model_path.c_str());

  coral::CameraStreamer streamer;
  const auto safety_input_path = absl::GetFlag(FLAGS_worker_safety_input);
  const auto visual_inspection_path = absl::GetFlag(FLAGS_visual_inspection_input);

  InferenceWrapper detector(detection_model_path, detection_label_path);
  size_t detector_input_size = detector.get_input_size();

  // Begins pipeline with a mixer for combining both streams.
  std::string pipeline = absl::StrFormat(
      "glvideomixer name=m sink_0::xpos=0 name=m "
      "sink_1::xpos=%d ! autovideosink name=overlaysink sync=false \n",
      width);

  // Begins pipelines with Worker Safety.
  pipeline += generate_pipeline_string(
      safety_input_path, width, height, detector_input_size, coral::kWorkerSafety);

  // Next, adds in the Visual Inspection.
  pipeline += generate_pipeline_string(
      visual_inspection_path, width, height, detector_input_size, coral::kVisualInspection);

  const gchar* kPipeline = pipeline.c_str();
  VLOG(2) << "Pipeline: " << pipeline.c_str();

  LOG(INFO) << "Starting Visual Inspection Demo\n";
  InferenceWrapper classifier(classifier_model_path, classifier_label_path);
  auto keepout_polygon = coral::parse_keepout_polygon(absl::GetFlag(FLAGS_keepout_points_path));
  streamer.run_pipeline(
      /*pipeline_string=*/kPipeline,
      /*safety_callback_data=*/
      {/*rsvg=*/nullptr, /*cb=*/
       [&](GstElement* rsvg, uint8_t* pixels, int pixel_length) {
         callback_helper::worker_safety_callback(
             rsvg, pixels, pixel_length, detector, width, height, threshold, keepout_polygon);
       }},
      /*inspection_callback_data=*/
      {/*rsvg=*/nullptr, /*cb=*/[&](GstElement* rsvg, uint8_t* pixels, int pixel_length) {
         callback_helper::visual_inspection_callback(
             rsvg, pixels, pixel_length, detector, classifier, width, height, threshold);
       }});
}
