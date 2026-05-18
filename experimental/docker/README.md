# Experimental Docker Build

This Dockerfile builds TensorRT Edge-LLM for Jetson Thor from the current
checkout using `nvcr.io/nvidia/pytorch:26.04-py3`. No build arguments are
required for the default Thor build.

```bash
docker build --network=host --shm-size=8g \
  -f experimental/docker/Dockerfile \
  -t tensorrt-edge-llm:experimental .
```

The build extracts the CuTe DSL tarball from `kernelSrcs/cuteDSLPrebuilt/` and
does not generate CuTe DSL artifacts during Docker build. In GitLab CI,
`l0_experimental_docker_jedha` downloads `build_cutedsl_sm110_artifact` first
and stages the tarball into that directory before sending the Docker context to
the board.

Run the OpenAI-compatible experimental server with the upstream Python module:

```bash
docker run --runtime nvidia --rm -it --network host \
  -v /data:/data \
  tensorrt-edge-llm:experimental \
  python3 -m experimental.server --model Qwen/Qwen3-1.7B
```

Model downloads and generated artifacts are cached under `/data`.
