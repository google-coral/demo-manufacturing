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

#ifndef MANUFACTURING_DEMO_IMAGE_UTILS_H_
#define MANUFACTURING_DEMO_IMAGE_UTILS_H_

#include <array>
#include <cstdint>
#include <vector>

#include "tensorflow/lite/interpreter.h"

namespace coral {
// Defines dimension of an image
using ImageDims = std::array<int, 3>;

// Defines a bounding box
struct BoundingBox {
  BoundingBox(int y1, int x1, int y2, int x2)
      : ymin(y1), xmin(x1), ymax(y2), xmax(x2), height(y2 - y1), width(x2 - x1) {}
  int ymin, xmin, ymax, xmax;
  int height, width;
};

// Crop an image
std::vector<uint8_t> crop_image(
    uint8_t* pixels, const ImageDims& image_dim, const BoundingBox& crop_area);

// Resize an image from in_dim to out_dim and return as a new vector
std::vector<uint8_t> resize_image(
    const uint8_t* in, const ImageDims& in_dim, const ImageDims& out_dims);

}  // namespace coral

#endif  // MANUFACTURING_DEMO_IMAGE_UTILS_H