#ifndef CAMERASTREAMER_H_
#define CAMERASTREAMER_H_

#include <glib.h>
#include <gst/gst.h>

#include <functional>
#include <memory>

#include "inference_wrapper.h"
#include "keepout_shape.h"

namespace coral {

class CameraStreamer {
public:
  CameraStreamer() = default;
  virtual ~CameraStreamer() = default;
  CameraStreamer(const CameraStreamer&) = delete;
  CameraStreamer& operator=(const CameraStreamer&) = delete;

  struct UserData {
    std::function<void(
        uint8_t* pixels, int length, GstElement* rsvg, InferenceWrapper& inferencer, uint16_t width,
        uint16_t height, float threshold, Polygon& keepout_polygon)>
        callback_func;
    GstElement* rsvg;  // handle to gsstreamer rsvgoverlay module, used by
                       // interpret_frame()
    InferenceWrapper& inferencer;
    uint16_t width, height;
    float threshold;
    Polygon& keepout_polygon;
  };

  void RunPipeline(const gchar* pipeline_string, UserData user_data);
};

}  // namespace coral

#endif  // CAMERASTREAMER_H_
