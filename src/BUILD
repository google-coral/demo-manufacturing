cc_library(
    name = "camera_streamer",
    srcs = ["camera_streamer.cc"],
    hdrs = ["camera_streamer.h"],
    deps = [
	    ":keepout_shape",
	    ":inference_wrapper",
            "@glog",
            "@system_libs//:gstreamer",
            "@system_libs//:gstallocators",
    ],
)

cc_library(
    name = "inference_wrapper",
    srcs = ["inference_wrapper.cc"],
    hdrs = ["inference_wrapper.h"],
    deps = [
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",
        "@glog",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "keepout_shape",
    srcs = ["keepout_shape.cc"],
    hdrs = ["keepout_shape.h"],
    deps = [
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings:str_format"
		    ],
)

cc_binary(
    name = "manufacturing_demo",
    srcs = ["manufacturing_demo.cc"],
    deps = [
        ":camera_streamer",
        ":inference_wrapper",
	":keepout_shape",
        "@glog",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings:str_format"
    ],
)
