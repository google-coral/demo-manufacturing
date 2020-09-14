cc_library(
    name = "edgetpu",
    deps = [
      "@libedgetpu//driver/beagle:beagle_all_driver_provider",
      "@libedgetpu//tflite/public:edgetpu",
      "@libedgetpu//tflite/public:edgetpu_c",
      "@libedgetpu//tflite:custom_op_direct",
    ],
)

cc_library(
    name = "camera_streamer",
    srcs = ["camera_streamer.cc"],
    hdrs = ["camera_streamer.h"],
    deps = [
            "@system_libs//:gstreamer",
            "@system_libs//:gstallocators",
    ],
)

cc_library(
    name = "inference_wrapper",
    srcs = ["inference_wrapper.cc"],
    hdrs = ["inference_wrapper.h"],
    deps = [
        ":edgetpu",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "manufacturing_demo",
    srcs = ["manufacturing_demo.cc"],
    deps = [
        ":camera_streamer",
        ":inference_wrapper",
        "@com_google_absl//absl/strings:str_format"
    ],
)
