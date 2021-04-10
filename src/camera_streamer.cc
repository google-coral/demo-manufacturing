#include "camera_streamer.h"

#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "glog/logging.h"

namespace coral {

namespace {

GstFlowReturn on_new_sample(GstElement* sink, void* data) {
  GstSample* sample;
  GstFlowReturn retval = GST_FLOW_OK;

  g_signal_emit_by_name(sink, "pull-sample", &sample);
  if (sample) {
    GstMapInfo info;
    auto buf = gst_sample_get_buffer(sample);
    if (gst_buffer_map(buf, &info, GST_MAP_READ) == TRUE) {
      auto cb_data = reinterpret_cast<CameraStreamer::CallbackData*>(data);
      // Pass the frame to the user callback
      cb_data->cb(cb_data->svg_gen, info.data, info.size);
    } else {
      LOG(ERROR) << "Couldn't get buffer info";
      retval = GST_FLOW_ERROR;
    }

    gst_buffer_unmap(buf, &info);
    gst_sample_unref(sample);
  }
  return retval;
}

gboolean on_bus_message(GstBus* bus, GstMessage* msg, gpointer data) {
  GMainLoop* loop = reinterpret_cast<GMainLoop*>(data);

  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS: {
      LOG(INFO) << "End of stream";
      g_main_loop_quit(loop);
      break;
    }
    case GST_MESSAGE_ERROR: {
      GError* error;
      gst_message_parse_error(msg, &error, nullptr);
      LOG(ERROR) << error->message;
      g_error_free(error);
      g_main_loop_quit(loop);
      break;
    }
    case GST_MESSAGE_WARNING: {
      GError* error;
      gst_message_parse_warning(msg, &error, nullptr);
      LOG(WARNING) << error->message;
      g_error_free(error);
      g_main_loop_quit(loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

}  // namespace

void CameraStreamer::prepare_appsink(
    GstElement* pipeline, const std::string name, CallbackData* callback_data) {
  // Set up an appsink to pass frames to a user callback
  auto appsink = gst_bin_get_by_name(
      reinterpret_cast<GstBin*>(pipeline), absl::StrCat("appsink_", name).c_str());
  CHECK_NOTNULL(appsink);

  g_object_set(appsink, "emit-signals", true, nullptr);
  g_signal_connect(
      appsink, "new-sample", reinterpret_cast<GCallback>(on_new_sample), callback_data);
}

void CameraStreamer::run_pipeline(
    const gchar* pipeline_string, CallbackData safety_callback_data,
    CallbackData inspection_callback_data) {
  gst_init(nullptr, nullptr);
  // Set up a pipeline based on the pipeline string
  auto loop = g_main_loop_new(nullptr, FALSE);
  CHECK_NOTNULL(loop);
  auto pipeline = gst_parse_launch(pipeline_string, nullptr);
  CHECK_NOTNULL(pipeline);

  GstElement* rsvg = gst_bin_get_by_name(GST_BIN(pipeline), "rsvg");
  CHECK_NOTNULL(rsvg);
  SvgGenerator svg_gen(rsvg);
  safety_callback_data.svg_gen = &svg_gen;
  inspection_callback_data.svg_gen = &svg_gen;

  // Prepare each appsink, ensuring the right frames reach their callback.
  prepare_appsink(pipeline, coral::kWorkerSafety, &safety_callback_data);
  prepare_appsink(pipeline, coral::kVisualInspection, &inspection_callback_data);

  // Add a bus watcher. It's safe to unref the bus immediately after
  auto bus = gst_element_get_bus(pipeline);
  CHECK_NOTNULL(bus);
  gst_bus_add_watch(bus, on_bus_message, loop);
  gst_object_unref(bus);

  // Start the pipeline, runs until interrupted, EOS or error
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(loop);

  // Cleanup
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
}

}  // namespace coral
