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

#include <glib.h>

#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"

namespace coral {

constexpr char* kSvgHeader = "<svg>";
constexpr char* kSvgFooter = "</svg>";
constexpr char* kSvgBox =
    "<rect x=\"$0\" y=\"$1\" width=\"$2\" height=\"$3\" "
    "fill-opacity=\"0.0\" "
    "style=\"stroke-width:5;stroke:rgb($4,$5,$6);\"/>";
constexpr char* kSvgText = "<text x=\"$0\" y=\"$1\" font-size=\"large\" fill=\"$2\">$3</text>";

class SvgGenerator {
public:
  SvgGenerator(GstElement* svg) : rsvg_(svg){};
  virtual ~SvgGenerator() = default;
  SvgGenerator(const SvgGenerator&) = delete;
  SvgGenerator& operator=(const SvgGenerator&) = delete;

  void set_worker_safety_svg(const std::string svg) LOCKS_EXCLUDED(lock_) {
    absl::MutexLock l(&lock_);
    worker_safety_svg_ = svg;
    update_svg();
  }
  void set_visual_inspection_svg(const std::string svg) LOCKS_EXCLUDED(lock_) {
    absl::MutexLock l(&lock_);
    visual_inspection_svg_ = svg;
    update_svg();
  }

private:
  void update_svg() EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    g_object_set(
        G_OBJECT(rsvg_), "data",
        absl::StrCat(kSvgHeader, worker_safety_svg_, visual_inspection_svg_, kSvgFooter).c_str(),
        NULL);
  }

  std::string worker_safety_svg_ GUARDED_BY(lock_) = "";
  std::string visual_inspection_svg_ GUARDED_BY(lock_) = "";
  GstElement* rsvg_ GUARDED_BY(lock_);
  // A mutex is needed to ensure the competing threads don't update the strings
  // before the SVG has been set in the overlay.
  absl::Mutex lock_;
};

}  // namespace coral
