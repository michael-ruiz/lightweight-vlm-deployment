# Lightweight VLM Deployment

Offline benchmark pipeline for evaluating lightweight Vision-Language Models (VLMs) on the [Drive&Act](https://driveandact.com/) driver-activity dataset, with first-class support for deployment on **NVIDIA Jetson Orin Nano (8 GB)**.

Supported models include [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) and [SmolVLM-256M](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct), with three inference backends: plain PyTorch, TensorRT-LLM, and **vLLM** (recommended for Jetson).

---

## Results

### Jetson Orin Nano (8 GB) — vLLM + AWQ

| Model | Backend | Resolution | Frames | Accuracy | Precision | FPS | VRAM |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B-AWQ | vLLM V0 | 504 × 504 | 3 | **54%** | **61%** | **1.15** | 3 582 MB |
| Qwen2.5-VL-3B-AWQ | vLLM V0 | 336 × 336 | 1 | 46% | 43% | 1.39 | 3 544 MB |

### Desktop PC — RTX 4090, PyTorch BF16

| Model | Backend | Resolution | Frames | Accuracy | Precision | FPS |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B | PyTorch | Full | 3 | **75%** | **88%** | 22 |

> **Gap analysis:** The ~21 pp accuracy difference is explained by AWQ INT4 quantization (~10 pp), resolution cap imposed by Jetson memory constraints (~8 pp), and different evaluation set sizes (50 vs. 12 stratified samples, ~3 pp).

---

## Hardware Requirements

| Tier | Device | Notes |
|---|---|---|
| Development | Any CUDA GPU with ≥ 8 GB VRAM | Full BF16, no constraints |
| Edge deployment | Jetson Orin Nano 8 GB | Requires vLLM backend + AWQ model — see [Jetson Notes](#jetson-orin-nano-deployment) |

---

## Dataset

Download the **Drive&Act** dataset from [driveandact.com](https://driveandact.com/) and place it under `data/`:

```
data/
└── Drive&Act/
    └── Drive&Act/
        └── kinect_color/
            ├── vp1/
            │   └── run1_*.mp4
            ├── vp2/
            ...
```

The loader automatically discovers `.mp4` segments and extracts frame-level labels from the filename/annotation metadata.

---

## Installation

```bash
pip install -r requirements.txt
```

For the **vLLM backend** (required on Jetson, optional on desktop):

```bash
# Standard PyPI — do NOT use the Jetson-specific index for these packages
pip install vllm scikit-learn pillow tqdm \
    --index-url https://pypi.org/simple
```

---

## Usage

### Quick smoke test (2 segments)

```bash
python3 run_benchmark.py \
  --backend vllm \
  --vllm-gpu-memory-utilization 0.85 \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
  --dataset-root data \
  --prompt-profile driveact \
  --frames-per-segment 1 \
  --limit 2 \
  --output smoke_test.json
```

### Full benchmark (50 segments, 3 frames each)

```bash
python3 run_benchmark.py \
  --backend vllm \
  --vllm-gpu-memory-utilization 0.85 \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
  --dataset-root data \
  --prompt-profile driveact \
  --frames-per-segment 3 \
  --limit 50 \
  --output benchmark_results.json
```

### Desktop (PyTorch BF16, no quantization)

```bash
python3 run_benchmark.py \
  --backend pytorch \
  --load-bits 0 \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset-root data \
  --prompt-profile driveact \
  --frames-per-segment 3 \
  --limit 50 \
  --output benchmark_desktop.json
```

### Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--backend` | `pytorch` | `pytorch` / `vllm` / `tensorrt-llm` |
| `--model-id` | SmolVLM-256M | Any HuggingFace VLM model ID |
| `--vllm-gpu-memory-utilization` | `0.9` | vLLM GPU memory fraction (use `0.85` on Jetson) |
| `--prompt-profile` | `driveact` | Prompt + label preset; auto-selects model-specific variant |
| `--frames-per-segment` | `3` | Frames sampled per segment; majority vote applied |
| `--limit` | all | Max segments to evaluate |
| `--load-bits` | `4` | `4`=NF4, `8`=INT8, `0`=FP16 (PyTorch backend only) |
| `--confidence-threshold` | `1.0` | Logit confidence gate for fallback label; `0.80` recommended |

---

## Project Structure

```
├── run_benchmark.py        # CLI entry point
├── requirements.txt
└── src/
    ├── data_loader.py      # Drive&Act video/frame loader
    ├── evaluator.py        # Benchmark loop, prompt profiles, metrics
    ├── model_engine.py     # PyTorch AutoModel inference engine
    ├── vllm_engine.py      # vLLM PagedAttention engine (Jetson-optimised)
    ├── trtllm_engine.py    # TensorRT-LLM engine stub
    └── hardware_monitor.py # VRAM, timing, and TPS measurement
```

---

## Prompt Profiles

The `--prompt-profile` flag selects a pre-tuned prompt and label set. Passing `driveact` (default) auto-selects the best sub-profile for the model:

| Profile | Auto-selected for | Labels |
|---|---|---|
| `driveact_qwen` | `Qwen2.5-VL-*`, `Qwen2-VL-*` | Texting, Drinking, Driving, Reaching |
| `driveact_smolvlm` | `SmolVLM-*` | Driving, Drinking, Reaching, Texting |
| `driveact` | All others | Driving, Texting, Drinking, Reaching |

---

## Jetson Orin Nano Deployment

The Jetson Orin Nano uses **NvMap**, a kernel-level memory manager that requires physically contiguous DRAM pages. Standard PyTorch `cudaMalloc` calls for large tensors (model weights, KV cache, activation buffers) frequently fail with `NvMapMemAllocInternalTagged: error 12` due to physical fragmentation.

### Recommended setup

1. **Switch to multi-user target** (free GPU memory from desktop):
   ```bash
   sudo systemctl isolate multi-user.target
   ```

2. **Use the AWQ-quantized model** — `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` occupies 3.29 GiB (vs. ~6 GiB for BF16).

3. **Use the vLLM backend** — PagedAttention allocates KV cache in small non-contiguous 16-token pages, bypassing the contiguous allocation requirement.

4. **Run the benchmark**:
   ```bash
   python3 run_benchmark.py \
     --backend vllm \
     --vllm-gpu-memory-utilization 0.85 \
     --model-id Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
     --dataset-root data \
     --prompt-profile driveact \
     --frames-per-segment 3 \
     --limit 50 \
     --output benchmark_results.json
   ```

### What the vLLM engine configures automatically

The engine (`src/vllm_engine.py`) applies the following settings at runtime to work around Jetson constraints — no manual configuration required:

| Setting | Value | Reason |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:128` | Prevents > 128 MB single cudaMalloc calls |
| `VLLM_USE_V1` | `0` | V1's profiler exhausts VRAM; V0 uses lighter heuristics |
| `enforce_eager` | `True` | Disables CUDA graphs (require large contiguous buffers) |
| `dtype` | `half` (FP16) | BF16 default needs more contiguous headroom |
| `num_gpu_blocks_override` | `64` | Caps KV cache at ~1.5 MB/layer (vs. 167 MB default) |
| `swap_space` | `0` | Disables 6.3 GB CPU KV swap that would exhaust RAM |
| `max_num_batched_tokens` | `512` | Limits profiling to 1 image item (not 57) |
| `max_model_len` | `512` | Sufficient for 324 image tokens + prompt + response |
| `max_pixels` | `504 × 504` | 2.25× more detail than minimum; stays within 512-token budget |

> **Do NOT** use `expandable_segments:True` on Jetson. It enables CUDA VMM (`cuMemCreate`/`cuMemMap`) which conflicts with NvMap and causes `NVML_SUCCESS == r INTERNAL ASSERT FAILED` crashes on every allocation.

---

## License

This project is released under the MIT License.
