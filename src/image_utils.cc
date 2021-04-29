// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "image_utils.h"

#include <fstream>

#include "glog/logging.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace coral {

std::vector<uint8_t> crop_image(
    uint8_t* pixels, const ImageDims& image_dim, const BoundingBox& crop_area) {
  std::vector<uint8_t> cropped_image;
  cropped_image.resize(crop_area.width * crop_area.height * image_dim[2]);

  uint8_t* src = nullptr;
  uint8_t* dst = &cropped_image[0];
  const int crop_width = crop_area.width * image_dim[2];
  for (int y = crop_area.ymin; y < crop_area.ymax; y++) {
    src = pixels + (y * image_dim[1] * image_dim[2]) + (crop_area.xmin * image_dim[2]);
    memcpy(dst, src, crop_width);
    dst += crop_width;
  }

  return cropped_image;
}

std::vector<uint8_t> resize_image(
    const uint8_t* in, const ImageDims& in_dims, const ImageDims& out_dims) {
  const int image_height = in_dims[0];
  const int image_width = in_dims[1];
  const int image_channels = in_dims[2];
  const int wanted_height = out_dims[0];
  const int wanted_width = out_dims[1];
  const int wanted_channels = out_dims[2];
  const int number_of_pixels = image_height * image_width * image_channels;
  std::unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter);
  int base_index = 0;
  // two inputs: input and new_sizes
  interpreter->AddTensors(2, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);
  // set input and output tensors
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});
  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input", {1, image_height, image_width, image_channels}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2}, quant);
  interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "output", {1, wanted_height, wanted_width, wanted_channels}, quant);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  const TfLiteRegistration* resize_op = resolver.FindOp(tflite::BuiltinOperator_RESIZE_BILINEAR, 1);
  auto* params =
      reinterpret_cast<TfLiteResizeBilinearParams*>(malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = false;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op, nullptr);
  interpreter->AllocateTensors();
  // fill input image
  // in[] are integers, cannot do memcpy() directly
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < number_of_pixels; i++) {
    input[i] = in[i];
  }
  // fill new_sizes
  interpreter->typed_tensor<int>(1)[0] = wanted_height;
  interpreter->typed_tensor<int>(1)[1] = wanted_width;
  interpreter->Invoke();

  std::vector<uint8_t> out;
  out.resize(out_dims[0] * out_dims[1] * out_dims[2]);
  auto output = interpreter->typed_tensor<float>(2);
  auto output_number_of_pixels = wanted_height * wanted_height * wanted_channels;
  for (int i = 0; i < output_number_of_pixels; i++) {
    out[i] = static_cast<uint8_t>(output[i]);
  }
  return out;
}

}  // namespace coral