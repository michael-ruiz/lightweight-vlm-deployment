"""Hardware monitoring and inference timing helpers.

The monitor is intentionally small and dependency-tolerant. It samples CUDA
memory in a background thread on desktop GPUs today, while leaving a clear hook
for Jetson-specific telemetry later.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

try:
    import pynvml
except ImportError:  # pragma: no cover - optional runtime dependency
    pynvml = None  # type: ignore[assignment]


BYTES_PER_MIB = 1024 * 1024


@dataclass(slots=True)
class GenerationTiming:
    """Per-generation timing summary."""

    ttft_seconds: Optional[float] = None
    total_seconds: float = 0.0
    generated_tokens: int = 0

    @property
    def tokens_per_second(self) -> float:
        """Return generated-token throughput over the measured decode window."""
        if self.generated_tokens <= 0 or self.total_seconds <= 0.0:
            return 0.0
        return self.generated_tokens / self.total_seconds


@dataclass(slots=True)
class InferenceTimer:
    """Small helper for TTFT and TPS measurement around model generation."""

    start_time: float = field(default_factory=time.perf_counter)
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    generated_tokens: int = 0

    def mark_first_token(self) -> None:
        """Record the first-token timestamp once."""
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def finish(self, generated_tokens: int) -> GenerationTiming:
        """Finalize and return a timing summary."""
        self.generated_tokens = max(0, generated_tokens)
        self.end_time = time.perf_counter()
        if self.first_token_time is None and self.generated_tokens > 0:
            self.first_token_time = self.end_time

        ttft = None
        if self.first_token_time is not None:
            ttft = self.first_token_time - self.start_time

        return GenerationTiming(
            ttft_seconds=ttft,
            total_seconds=self.end_time - self.start_time,
            generated_tokens=self.generated_tokens,
        )


class HardwareMonitor:
    """Background sampler for CUDA memory and inference timing helpers."""

    def __init__(self, sample_interval_seconds: float = 0.25, cuda_device_index: int = 0) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self.cuda_device_index = cuda_device_index
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._peak_vram_mb = 0.0
        self._nvml_handle = None
        self._nvml_initialized = False

    def start(self) -> None:
        """Start background sampling."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._init_nvml()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="hardware-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop background sampling and release optional NVML state."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.sample_interval_seconds * 4))
        if self._nvml_initialized and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
        self._nvml_initialized = False
        self._nvml_handle = None

    def create_timer(self) -> InferenceTimer:
        """Create a timer for one generation call."""
        return InferenceTimer()

    @staticmethod
    def get_device_info() -> dict:
        """Return a snapshot of CUDA/GPU availability for report self-auditing."""
        cuda_available = torch.cuda.is_available()
        info: dict = {"cuda_available": cuda_available}
        if cuda_available:
            try:
                props = torch.cuda.get_device_properties(0)
                info["gpu_name"] = props.name
                info["gpu_total_vram_mb"] = round(props.total_memory / BYTES_PER_MIB, 1)
                info["cuda_capability"] = f"{props.major}.{props.minor}"
            except RuntimeError:
                pass
            try:
                info["cuda_version"] = torch.version.cuda  # type: ignore[attr-defined]
            except AttributeError:
                pass
        else:
            info["gpu_name"] = None
        return info

    def update_peak_vram(self) -> None:
        """Synchronously sample VRAM once and update the peak."""
        usage_mb = self._current_vram_mb()
        with self._lock:
            self._peak_vram_mb = max(self._peak_vram_mb, usage_mb)

    @property
    def peak_vram_mb(self) -> float:
        """Highest sampled CUDA memory use in MiB."""
        with self._lock:
            return self._peak_vram_mb

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self.update_peak_vram()
            self._stop_event.wait(self.sample_interval_seconds)
        self.update_peak_vram()

    def _current_vram_mb(self) -> float:
        if torch.cuda.is_available():
            try:
                reserved = torch.cuda.memory_reserved(self.cuda_device_index) / BYTES_PER_MIB
            except RuntimeError:
                reserved = 0.0

            nvml_used = self._current_nvml_used_mb()
            return max(reserved, nvml_used)

        return 0.0

    def _init_nvml(self) -> None:
        if pynvml is None or not torch.cuda.is_available():
            return
        try:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.cuda_device_index)
            self._nvml_initialized = True
        except pynvml.NVMLError:
            self._nvml_initialized = False
            self._nvml_handle = None

    def _current_nvml_used_mb(self) -> float:
        if pynvml is None or not self._nvml_initialized or self._nvml_handle is None:
            return 0.0
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        except pynvml.NVMLError:
            return 0.0
        return float(info.used) / BYTES_PER_MIB

    # Jetson extension point:
    # Add jtop-backed sampling here for power, RAM, GPU load, and temperature.
    # Keep imports local to that future implementation so CUDA desktop runs do
    # not require Jetson-only packages.


__all__ = ["GenerationTiming", "HardwareMonitor", "InferenceTimer"]

