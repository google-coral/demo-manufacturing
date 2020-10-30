# Manufacturing Demo

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

## Quick run

First, download example models using the provided script

```
sh download_models.sh
```

Run without collision detection:

```
./out/$ARCH/demo/manufacturing_demo
```

Run with an example collision detection config:
```
./out/k8/demo/manufacturing_demo --keepout_points_path config/keepout_points.csv
```
