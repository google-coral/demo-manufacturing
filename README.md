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


