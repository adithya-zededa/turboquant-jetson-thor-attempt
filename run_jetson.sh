#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# TurboQuant — Jetson AGX Thor runner
#
# Usage:
#   ./run_jetson.sh            # build image + run demo
#   ./run_jetson.sh demo       # same as above
#   ./run_jetson.sh shell      # drop into container shell
#   ./run_jetson.sh build      # build image only
#   ./run_jetson.sh clean      # remove local image
#
# The image is built from Dockerfile.jetson which extends:
#   nvcr.io/nvidia/tritonserver:26.03-vllm-python-py3
# (PyTorch 2.11, Triton 3.6, CUDA — already on the Thor)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

IMAGE="turboquant-jetson:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD="${1:-demo}"

build_image() {
    echo "==> Building $IMAGE ..."
    docker build \
        --file  "$SCRIPT_DIR/Dockerfile.jetson" \
        --tag   "$IMAGE" \
        "$SCRIPT_DIR"
    echo "==> Build complete: $IMAGE"
}

run_container() {
    local entrypoint_cmd="${1:-python3 /workspace/demo_jetson.py}"
    docker run --rm \
        --runtime nvidia \
        --gpus all \
        --network host \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e TOKENIZERS_PARALLELISM=false \
        ${MODEL:+-e MODEL="$MODEL"} \
        -v "$SCRIPT_DIR/turboquant:/workspace/turboquant" \
        -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
        "$IMAGE" \
        bash -c "$entrypoint_cmd"
}

case "$CMD" in
    build)
        build_image
        ;;
    demo)
        build_image
        echo ""
        echo "==> Running TurboQuant demo on Jetson AGX Thor ..."
        run_container "python3 /workspace/demo_jetson.py"
        ;;
    vllm)
        build_image
        echo ""
        echo "==> Running TurboQuant + vLLM inference on Jetson AGX Thor ..."
        echo "    Model: ${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
        run_container "python3 /workspace/demo_vllm.py"
        ;;
    v8)
        build_image
        echo ""
        echo "==> Running TurboQuant v8 benchmark on Jetson AGX Thor ..."
        echo "    Model: ${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
        run_container "python3 /workspace/demo_v8.py"
        ;;
    v8-kernels)
        build_image
        echo ""
        echo "==> Running TurboQuant v8 kernel-only benchmarks (no vLLM) ..."
        run_container "SKIP_VLLM=1 python3 /workspace/demo_v8.py"
        ;;
    shell)
        build_image
        echo "==> Starting interactive shell ..."
        docker run --rm -it \
            --runtime nvidia \
            --gpus all \
            --network host \
            -e CUDA_VISIBLE_DEVICES=0 \
            -e TOKENIZERS_PARALLELISM=false \
            -v "$SCRIPT_DIR/turboquant:/workspace/turboquant" \
            -v "$SCRIPT_DIR:/workspace/host" \
            "$IMAGE" \
            /bin/bash
        ;;
    clean)
        echo "==> Removing $IMAGE ..."
        docker rmi "$IMAGE" 2>/dev/null && echo "Removed." || echo "Image not found."
        ;;
    *)
        echo "Unknown command: $CMD"
        echo "Usage: $0 {build|demo|vllm|v8|v8-kernels|shell|clean}"
        exit 1
        ;;
esac
