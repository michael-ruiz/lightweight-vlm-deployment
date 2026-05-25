# realsense-jetson-vlm

Open-source scaffold for benchmarking and live deployment of small quantized
vision-language models on NVIDIA Jetson edge devices for real-time driver
action recognition.

The initial sensor target is an Intel RealSense D435 using the RGB stream at
640x480 and 30 FPS. The camera layer is structured so the IR imager can be
enabled later from configuration. The evaluation target is extracted Drive&Act
frame data mapped into a five-class automotive safety taxonomy:

`Driving`, `Texting`, `Drinking`, `Reaching`, `Asleep`.

## Repository Layout

```text
config/
  config.yaml
src/
  __init__.py
  camera.py
  inference.py
  evaluator.py
main_live.py
main_benchmark.py
requirements.txt
README.md
```

## Setup

Jetson Python environments often require NVIDIA-provided `torch` wheels and a
RealSense SDK build that matches the board image. Install those platform
packages first, then install the remaining Python dependencies:

```bash
python -m pip install -r requirements.txt
```

If `torch`, `bitsandbytes`, or `pyrealsense2` are not available as standard
wheels for your Jetson image, install the Jetson-compatible builds manually and
then rerun the requirements install for the rest of the stack.

## Configuration

Edit `config/config.yaml`.

Model selection:

```yaml
model:
  selected: smolvlm_256m
```

Supported selectors:

- `qwen2_5_vl_3b`: `Qwen/Qwen2.5-VL-3B-Instruct`
- `smolvlm_256m`: `HuggingFaceTB/SmolVLM-256M-Instruct`
- `gemma_3_4b`: `google/gemma-3-4b-it`

All models are loaded with 4-bit NF4 quantization and `torch.float16` compute
dtype to reduce unified memory pressure on Jetson.

Camera defaults:

```yaml
camera:
  stream: rgb
  width: 640
  height: 480
  fps: 30
```

IR capture is scaffolded under `camera.ir` and disabled by default.

## Live Inference

Connect the RealSense D435 and run:

```bash
python main_live.py --config config/config.yaml
```

The live loop starts a dedicated camera thread and keeps only the newest frame,
dropping unread frames so inference never accumulates stale images. The console
prints the predicted driver state, measured inference FPS, latency, and
time-to-first-token when streaming decode is available.

Stop with `Ctrl+C`; the RealSense pipeline is closed in a `finally` block.

## Drive&Act Benchmarking

Extract Drive&Act frames into a directory tree where the action name appears in
the folder or filename. Then set:

```yaml
benchmark:
  dataset_root: /path/to/drive_and_act/extracted_frames
  label_mapping:
    texting: Texting
    drinking: Drinking
```

The mapping is intentionally YAML-driven because Drive&Act exports and local
preprocessing layouts vary. Matching is case-insensitive and searches the
relative path of each image.

Run:

```bash
python main_benchmark.py --config config/config.yaml
```

The benchmark reports:

- Accuracy
- Macro precision
- Macro recall
- Average inference latency
- Average time-to-first-token, when available

Per-frame predictions are written to `benchmark.output_csv` when configured.

## Prompt

The strict prompt in `config/config.yaml` is:

```text
You are an automotive safety system. Analyze the driver's action in this image. Reply with ONLY ONE of the following words: [Driving, Texting, Drinking, Reaching, Asleep].
```

Generated text is normalized back to the configured labels. Outputs that cannot
be matched are reported as `Unknown`.

## Smoke Checks

Run static checks without requiring camera hardware:

```bash
python -m py_compile main_live.py main_benchmark.py src/*.py
python - <<'PY'
from src.camera import CameraStreamer
from src.inference import VLMInferenceEngine
from src.evaluator import discover_driveact_frames
print("imports ok")
PY
```

The import check does not instantiate the VLM, so it does not download models or
require CUDA.
