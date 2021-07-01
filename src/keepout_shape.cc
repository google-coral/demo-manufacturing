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

#include "keepout_shape.h"

#include <iostream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"

ABSL_FLAG(
    bool, safety_check_whole_box, false,
    "The worker safety demo should check for collisions on the entire "
    "detected box (better for cameras overhead) over just checking the "
    "bottom (for high-angle/diagonal feeds).");

namespace coral {

const double Point::get_distance(const Point& p) const {
  return sqrt(pow((x_ - p.x_), 2) + pow((y_ - p.y_), 2));
}
const int Point::get_direction(const Point& b, const Point& c) const {
  const int cross_product = (b.y_ - y_) * (c.x_ - b.x_) - (b.x_ - x_) * (c.y_ - b.y_);
  if (cross_product == 0) {
    return 0;
  }
  return (cross_product > 0) ? 1 : 2;
}

bool Line::contains_point(const Point& p) const {
  // The distance between begin_ to p + p to end_ should equal to length of
  // this line: begin_----------p-----end_
  return ((begin_.get_distance(p) + end_.get_distance(p)) - length_) < EPSILON;
}
bool Line::intersects_line(const Line& l) const {
  int dir1 = begin_.get_direction(end_, l.begin_);
  int dir2 = begin_.get_direction(end_, l.end_);
  int dir3 = l.begin_.get_direction(l.end_, begin_);
  int dir4 = l.begin_.get_direction(l.end_, end_);
  if (dir1 != dir2 && dir3 != dir4) return true;  // These 2 lines are intersecting.
  if (dir1 == 0 && contains_point(l.begin_)) return true;
  if (dir2 == 0 && contains_point(l.end_)) return true;
  if (dir3 == 0 && l.contains_point(begin_)) return true;
  if (dir4 == 0 && l.contains_point(end_)) return true;
  return false;
}
const std::string Line::info() const {
  return "((" + std::to_string(begin_.x_) + "," + std::to_string(begin_.y_) + "),("
         + std::to_string(end_.x_) + "," + std::to_string(end_.y_) + "))";
}

Polygon::Polygon(std::vector<Point>& polygon_points) {
  lines_.emplace_back(polygon_points[0], polygon_points[polygon_points.size() - 1]);
  for (int i = 0; i < polygon_points.size() - 1; i++)
    lines_.emplace_back(polygon_points[i], polygon_points[i + 1]);
}

Box::Box(int x1, int y1, int x2, int y2) {
  points_.emplace_back(x1, y1);
  points_.emplace_back(x1, y2);
  points_.emplace_back(x2, y1);
  points_.emplace_back(x2, y2);
  lines_.emplace_back(points_[0], points_[1]);
  lines_.emplace_back(points_[1], points_[2]);
  lines_.emplace_back(points_[2], points_[3]);
  lines_.emplace_back(points_[3], points_[0]);
  bottom_y_ = std::max(y1, y2);
}
const bool Box::intersects_line(const Line& l) const {
  for (const auto& line : lines_) {
    if (line.intersects_line(l)) {
      return true;
    }
  }
  return false;
}
const bool Box::collided_with_polygon(const Polygon& p, const uint32_t max_width) const {
  const auto& polygon_lines = p.get_lines();
  for (const auto& p : points_) {
    // If not checking the whole box, ignore points that aren't on the
    // bottom of the box.
    if (!absl::GetFlag(FLAGS_safety_check_whole_box) && p.y_ != bottom_y_) {
      continue;
    }
    size_t intersect_time{0};
    // We create a horizontal line from this point to max image width, if
    // it interects the lines in the polygon even time, it is not inside
    // the polygon. If it is odd, it is inside the polygon.
    Line extreme{p, {max_width, p.y_}};
    for (const auto& l : polygon_lines) {
      // If p is on any of the lines, it is a collision.
      if (l.contains_point(p)) {
        return true;
      }
      // If this line in the polygon intersects the extreme line.
      if (l.intersects_line(extreme)) {
        intersect_time++;
      }
    }
    if (intersect_time % 2 == 1) {
      return true;
    }
  }
  // Check for line intersection between this box and the polygon.
  for (const auto& box_line : lines_) {
    // If not checking the whole box, skip all lines that don't have the y
    // coordinate equal to the bottom of the box.
    if (!absl::GetFlag(FLAGS_safety_check_whole_box)
        && (box_line.begin_.y_ != bottom_y_ || box_line.end_.y_ != bottom_y_)) {
      continue;
    }
    for (const auto& polygon_line : p.get_lines()) {
      if (box_line.intersects_line(polygon_line)) {
        return true;
      }
    }
  }
  return false;
}

Polygon parse_keepout_polygon(const std::string& file_path) {
  std::ifstream f{file_path};
  std::vector<Point> points;
  if (f.is_open()) {
    std::string polygon_svg;
    polygon_svg = "<polygon points=\"";
    // Ignores csv header.
    f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    for (std::string line; std::getline(f, line);) {
      polygon_svg = absl::StrCat(polygon_svg, " ", line);
      int x, y;
      std::vector<std::string> p = absl::StrSplit(line, ',');
      absl::SimpleAtoi(p[0], &x);
      absl::SimpleAtoi(p[1], &y);
      points.emplace_back(x, y);
    }
    polygon_svg =
        absl::StrCat(polygon_svg, " \" style=\"fill:none;stroke:red;stroke-width:5\" /> ");
    Polygon keepout_polygon(points);
    keepout_polygon.set_svg_str(polygon_svg);
    return keepout_polygon;
  }
  return {};
}

}  // namespace coral
