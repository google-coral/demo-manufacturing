#ifndef CAMERASTREAMER_H_
#define CAMERASTREAMER_H_

#include <glib.h>
#include <gst/gst.h>

#include <functional>

namespace coral {

class CameraStreamer {
 public:
  CameraStreamer() = default;
  virtual ~CameraStreamer() = default;
  CameraStreamer(const CameraStreamer &) = delete;
  CameraStreamer &operator=(const CameraStreamer &) = delete;

  struct UserData {
    std::function<void(uint8_t *pixels, int length, GstElement *rsvg,
                       void *args)>
        f;
    GstElement *rsvg;  // handle to gsstreamer rsvgoverlay module, used by
                       // interpret_frame()
    void *args;
  };

  void RunPipeline(const gchar *pipeline_string, UserData user_data);
};

}  // namespace coral

#endif  // CAMERASTREAMER_H_
