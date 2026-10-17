#!/bin/sh
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV="$APP_DIR/.venv"
SOURCE="$VENV/llama.cpp"
RUNTIME="$VENV/llama-runtime"
IMAGE_SOURCE="$VENV/stable-diffusion.cpp"
IMAGE_BUILD="$IMAGE_SOURCE/build-command-center-metal"
IMAGE_RUNTIME="$IMAGE_BUILD/bin/sd-cli"
MLX_VENV="$VENV/mlx-runtime"

cd "$APP_DIR"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c 'import pypdf, reportlab' >/dev/null 2>&1; then
  echo "Setting up local PDF tools…"
  "$VENV/bin/python" -m pip install -r "$APP_DIR/requirements.txt"
fi

if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  if [ ! -x "$MLX_VENV/bin/python" ]; then
    if ! command -v python3.11 >/dev/null 2>&1; then
      echo "Apple MLX models need Python 3.11. Install Python 3.11, then open this app again."
      exit 1
    fi
    echo "Setting up the in-app Apple MLX model engine (one-time download)…"
    python3.11 -m venv "$MLX_VENV"
  fi
  if ! "$MLX_VENV/bin/python" -c 'import mlx_lm' >/dev/null 2>&1; then
    echo "Installing Apple MLX model support…"
    "$MLX_VENV/bin/python" -m pip install --upgrade mlx-lm
  fi
fi

NEED_CPU=0
NEED_METAL=0
[ -x "$RUNTIME/cpu/llama-server" ] || NEED_CPU=1
if [ "$(uname -m)" = "arm64" ]; then
  [ -x "$RUNTIME/metal/llama-server" ] || NEED_METAL=1
fi

if [ "$NEED_CPU" -eq 1 ] || [ "$NEED_METAL" -eq 1 ]; then
  if ! command -v git >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1; then
    echo "The local model engine needs Git and CMake. Install Apple's Command Line Tools, then open this app again."
    exit 1
  fi

  if [ ! -d "$SOURCE/.git" ]; then
    echo "Setting up the in-app model engine (one-time download)…"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$SOURCE"
  fi

  if [ "$NEED_CPU" -eq 1 ]; then
    echo "Building the in-app model engine for CPU…"
    cmake -S "$SOURCE" -B "$SOURCE/build-command-center-cpu" \
      -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$SOURCE/build-command-center-cpu" --config Release --target llama-server --parallel 4
    mkdir -p "$RUNTIME/cpu"
    cp -R "$SOURCE/build-command-center-cpu/bin/." "$RUNTIME/cpu/"
  fi

  if [ "$NEED_METAL" -eq 1 ]; then
    echo "Building the in-app model engine for Apple GPU…"
    if cmake -S "$SOURCE" -B "$SOURCE/build-command-center-metal" \
      -DGGML_METAL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release \
      && cmake --build "$SOURCE/build-command-center-metal" --config Release --target llama-server --parallel 4; then
      mkdir -p "$RUNTIME/metal"
      cp -R "$SOURCE/build-command-center-metal/bin/." "$RUNTIME/metal/"
    else
      echo "Apple GPU setup did not finish. The CPU model engine is ready and will be used instead."
    fi
  fi
fi

if [ ! -x "$IMAGE_RUNTIME" ]; then
  if ! command -v git >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1; then
    echo "The local image engine needs Git and CMake. Install Apple's Command Line Tools, then open this app again."
    exit 1
  fi
  if [ ! -d "$IMAGE_SOURCE/.git" ]; then
    echo "Setting up the in-app image engine (one-time download)…"
    git clone --depth 1 --recursive https://github.com/leejet/stable-diffusion.cpp.git "$IMAGE_SOURCE"
  fi
  echo "Building the in-app image engine for Apple GPU…"
  if cmake -S "$IMAGE_SOURCE" -B "$IMAGE_BUILD" \
    -DSD_METAL=ON -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_BUILD_EXAMPLES=ON \
    -DGGML_OPENMP=OFF -DGGML_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
    && cmake --build "$IMAGE_BUILD" --config Release --target sd-cli --parallel 4; then
    :
  else
    echo "Apple GPU image setup did not finish. Building the in-app CPU image engine instead…"
    IMAGE_BUILD="$IMAGE_SOURCE/build-command-center-cpu"
    cmake -S "$IMAGE_SOURCE" -B "$IMAGE_BUILD" \
      -DSD_METAL=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_BUILD_EXAMPLES=ON \
      -DGGML_OPENMP=OFF -DGGML_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build "$IMAGE_BUILD" --config Release --target sd-cli --parallel 4
    mkdir -p "$IMAGE_SOURCE/build-command-center-metal/bin"
    cp "$IMAGE_BUILD/bin/sd-cli" "$IMAGE_SOURCE/build-command-center-metal/bin/sd-cli"
  fi
fi

exec "$VENV/bin/python" "$APP_DIR/app/server.py"
