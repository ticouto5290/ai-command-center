FROM debian:bookworm-slim AS runtime-builder

ARG LLAMA_CPP_REF=23f7aabc16c72f6f707dba227251c35d7bd1ed96
ARG STABLE_DIFFUSION_CPP_REF=68285df319457e2d79310b48513e4ba1bf8ebe9a

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake git pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN git init /src/llama.cpp && \
    git -C /src/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git && \
    git -C /src/llama.cpp fetch --depth 1 origin "$LLAMA_CPP_REF" && \
    git -C /src/llama.cpp checkout --detach FETCH_HEAD && \
    git -C /src/llama.cpp submodule update --init --recursive --depth 1
RUN cmake -S /src/llama.cpp -B /src/llama.cpp/build-command-center-cpu \
      -DGGML_METAL=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=ON \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /src/llama.cpp/build-command-center-cpu --config Release \
      --target llama-server --parallel 2 && \
    mkdir -p /out/llama-runtime/cpu && \
    cp -a /src/llama.cpp/build-command-center-cpu/bin/. /out/llama-runtime/cpu/

RUN git init /src/stable-diffusion.cpp && \
    git -C /src/stable-diffusion.cpp remote add origin https://github.com/leejet/stable-diffusion.cpp.git && \
    git -C /src/stable-diffusion.cpp fetch --depth 1 origin "$STABLE_DIFFUSION_CPP_REF" && \
    git -C /src/stable-diffusion.cpp checkout --detach FETCH_HEAD && \
    git -C /src/stable-diffusion.cpp submodule update --init --recursive --depth 1
RUN cmake -S /src/stable-diffusion.cpp -B /src/stable-diffusion.cpp/build-command-center-cpu \
      -DSD_METAL=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_BUILD_EXAMPLES=ON \
      -DGGML_OPENMP=ON -DGGML_NATIVE=OFF -DGGML_BUILD_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /src/stable-diffusion.cpp/build-command-center-cpu --config Release \
      --target sd-cli --parallel 2 && \
    mkdir -p /out/stable-diffusion.cpp/build-command-center-metal/bin && \
    cp -a /src/stable-diffusion.cpp/build-command-center-cpu/bin/. \
      /out/stable-diffusion.cpp/build-command-center-metal/bin/

FROM python:3.11-slim-bookworm AS app

ENV PYTHONUNBUFFERED=1 \
    AICC_HOST=0.0.0.0 \
    AICC_PORT=8765 \
    AICC_LM_STUDIO_MODELS=/models/lmstudio \
    LD_LIBRARY_PATH=/opt/command-center/.venv/llama-runtime/cpu:/opt/command-center/.venv/stable-diffusion.cpp/build-command-center-metal/bin

WORKDIR /opt/command-center
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /opt/command-center/.venv/llama-runtime/cpu \
      /opt/command-center/.venv/stable-diffusion.cpp/build-command-center-metal/bin \
      /opt/command-center/app/data

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY README.md ./README.md
COPY --from=runtime-builder /out/llama-runtime/cpu/ ./.venv/llama-runtime/cpu/
COPY --from=runtime-builder /out/stable-diffusion.cpp/build-command-center-metal/bin/ \
  ./.venv/stable-diffusion.cpp/build-command-center-metal/bin/

EXPOSE 8765
CMD ["python", "app/server.py"]
