# Coral Manufacturing Demo

The Coral Manufacturing Demo demonstrates how to use a Coral Edge TPU to accomplish two common manufacturing use-cases:

* Worker Safety
* Visual Inspection

With a single Coral Edge TPU, it is possible to run both applications in real time! The demo includes example videos designed for the specific application, but camera sources and other videos are accepted. The demo requires OpenGL and is intended for either x86 platforms or the [Coral Dev Board](https://coral.ai/products/dev-board/) but building for x86, ARM64, and ARMv7 is all supported.

This demo is able to achieve realtime processing and display by taking advantage of the following features:

* [Model Cocompilation](https://coral.ai/products/dev-board/) - The two models are compiled together, allowing the TPU cache to store both model parameters. This lets you switch between models without needing to reload.
* Cascaded Models - For Visual Inspection, the output bounding box of the MobileDet detection is used to crop an image for the quality classifier. While in this demo only one apple is on a screen at a time, this approach can allow visual inspection of multiple objects in a frame.
* Retraining (via Keras API) - The apple quality classifier shows how to use the Keras API to retrain a known model. More info can be found in the [retraining script](models/retraining/train_classifier.py).
* OpenGL Acceleration - In order to mix the two demos, OpenGL is used. Also included is a new GStreamer element, glsvgoverlaysink, that allows generic systems to add a SVG overlay to an existing GL surface. This improves latency by not requiring a conversion to CPU for running [rsvgoverlay](https://gstreamer.freedesktop.org/documentation/rsvg/rsvgoverlay.html).

## Demo Overview

**Worker Safety**

For worker safety, a keepout region is defined (from a [CSV file](config/keepout_points.csv)) and MobileDet (trained on COCO17 dataset) is run to detect people. When a person is outside of this region the bounding box is green, if they are within it is red. The algorithm for determining collisions can be found in [keepout_shape.cc](src/keepout_shape.cc).

The default video is taken from [this repo](https://github.com/intel-iot-devkit/sample-videos).

**Visual Inspection**

The visual inspection demo grades the quality of apples on a conveyor belt. This is accomplished by cascading two models - MobileDet detection (for detecting an apple) and then cropping to the apple and running MobileNet v2 classification to determine if the apple is good or rotten. The detection model is the standard COCO17 detection available on the [Coral Model Page](https://coral.ai/models/), but the classification model required retraining. 

For classification, this [fresh/rotten fruit dataset](https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification) was used (taking only the apples). Keras API is used to load the base MobileNet V2 model and retain the [model](models/retraining/classifier.h5). TFlite APIs are then used for post-training quantization and conversion to [CPU TFLite](modles/retraining/classifier.tflite). Finally, the model is run through the EdgeTPU compiler with the MobileDet model to cocompile. A script for running this process on your own can be found [here](models/retraining/train_classifier.py).

## Compiling the demo

You will need to install docker on your computer first to compile the project.

If you want to build for all CPU architectures (amd64, arm64, armv7):

```
make DOCKER_TARGETS=demo docker-build
```

For building for a specific architecture, provide DOCKER_CPUS:

```
make DOCKER_TARGETS=demo DOCKER_CPUS=aarch64 docker-build
```

The binary should be in `out/$ARCH/demo` directory.

## Run the Demo

The default options will run the demo with the two example videos, a default keepout region, and the two cocompiled models (MobileDet and MobileNet V2).

```
./out/$ARCH/demo/manufacturing_demo
```

