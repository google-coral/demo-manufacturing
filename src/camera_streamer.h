#ifndef MANUFACTURING_DEMO_CAMERASTREAMER_H_
#define MANUFACTURING_DEMO_CAMERASTREAMER_H_

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
  // handle to gstreamer rsvgoverlay module, used by callbacks
  struct CallbackData {
    GstElement* rsvg;
    std::function<void(GstElement*, uint8_t*, int)> cb;
  };
  // Run pipeline with userdata and a callback fundtion
  void run_pipeline(const gchar* pipeline_string, CallbackData callback_data);
};

}  // namespace coral

#endif  // MANUFACTURING_DEMO_CAMERASTREAMER_H_
