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