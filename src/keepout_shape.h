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

#ifndef MANUFACTURING_DEMO_SHAPE_H_
#define MANUFACTURING_DEMO_SHAPE_H_

#include <math.h>

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace coral {
constexpr double EPSILON = 1E-9;

struct Point {
  Point() : x_(0), y_(0) {}
  Point(int x, int y) : x_(x), y_(y) {}
  Point(const Point& p) : x_(p.x_), y_(p.y_) {}
  Point(const std::pair<int, int>& p) : x_(p.first), y_(p.second) {}
  // Return distance between this point and the point p.
  const double get_distance(const Point& p) const;
  // Get the direction for this point and point b, c.
  // Return value is an int with value:
  // 0 if the three points is collinear
  // 1 if the direction is clockwise
  // 2 if the direction is counter clockwise
  const int get_direction(const Point& b, const Point& c) const;
  int x_;
  int y_;
};

struct Line {
  Line(const Point& begin, const Point& end)
      : begin_(begin), end_(end), length_(begin.get_distance(end)) {}
  // Checks whether if this line contains the point p.
  bool contains_point(const Point& p) const;
  // Checks whether if this line intersects the line l.
  bool intersects_line(const Line& l) const;
  const std::string info() const;
  Point begin_, end_;
  double length_;
};

class Polygon {
public:
  Polygon() {}
  Polygon(std::vector<Point>& polygon_points);
  // Return a vector of lines in this polygon.
  const std::vector<Line>& get_lines() const { return lines_; }
  // Return a reference to the string representing this polygon in svg
  // form.
  const std::string& get_svg_str() const { return svg_str_; };
  // Set svg string.
  void set_svg_str(const std::string& svg) { svg_str_ = svg; };

private:
  std::vector<Line> lines_;
  std::string svg_str_{"None"};
};

class Box {
public:
  Box(const int x1, const int y1, const int x2, const int y2);
  // Return true if this box collided with the polygon p.
  const bool collided_with_polygon(const Polygon& p, const uint32_t max_width) const;
  // Return true if this box collided with the line l.
  const bool intersects_line(const Line& l) const;
  const std::string info() const;

private:
  std::vector<Point> points_;
  std::vector<Line> lines_;
};

Polygon parse_keepout_polygon(const std::string& file_path);

}  // namespace coral

#endif  // MANUFACTURING_DEMO_SHAPE_H_
