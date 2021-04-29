/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MANUFACTURING_DEMO_CAMERASTREAMER_H_
#define MANUFACTURING_DEMO_CAMERASTREAMER_H_

#include <glib.h>
#include <gst/gst.h>

#include <functional>
#include <memory>

#include "inference_wrapper.h"
#include "keepout_shape.h"
#include "svg_generator.h"

namespace coral {

const std::string kVisualInspection = "inspection";
const std::string kWorkerSafety = "safety";

class CameraStreamer {
public:
  CameraStreamer() = default;
  virtual ~CameraStreamer() = default;
  CameraStreamer(const CameraStreamer&) = delete;
  CameraStreamer& operator=(const CameraStreamer&) = delete;
  // handle to gstreamer rsvgoverlay module, used by callbacks
  struct CallbackData {
    SvgGenerator* svg_gen;
    std::function<void(SvgGenerator*, uint8_t*, int)> cb;
  };
  // Run pipeline with userdata and a callback function.
  void run_pipeline(
      const gchar* pipeline_string, CallbackData safety_callback_data,
      CallbackData inspection_callback_data);

private:
  void prepare_appsink(GstElement* pipeline, const std::string name, CallbackData* callback_data);
};

}  // namespace coral

#endif  // MANUFACTURING_DEMO_CAMERASTREAMER_H_
