"""Drive&Act frame discovery, label parsing, and image/video preprocessing."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps


LOGGER = logging.getLogger(__name__)

CANONICAL_LABELS: tuple[str, ...] = ("Driving", "Texting", "Drinking", "Reaching", "Asleep")
LABEL_ALIASES: dict[str, str] = {
    "drive": "Driving",
    "driving": "Driving",
    "text": "Texting",
    "texting": "Texting",
    "phone": "Texting",
    "drink": "Drinking",
    "drinking": "Drinking",
    "reach": "Reaching",
    "reaching": "Reaching",
    "asleep": "Asleep",
    "sleep": "Asleep",
    "sleeping": "Asleep",
    "drowsy": "Asleep",
}
DRIVEACT_ACTIVITY_TO_LABEL: dict[str, str] = {
    "sitting_still": "Driving",
    "interacting_with_phone": "Texting",
    "talking_on_phone": "Texting",
    "drinking": "Drinking",
    "eat_drink": "Drinking",
    "fetching_an_object": "Reaching",
    "placing_an_object": "Reaching",
    "looking_or_moving_around (e.g. searching)": "Reaching",
    "reaching_for": "Reaching",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True, slots=True)
class FrameSample:
    """One preprocessed image frame and its inferred ground-truth label."""

    image: Image.Image
    label: str
    path: Path


class DriveAndActLoader:
    """Stream Drive&Act samples from extracted images or annotated MP4 files.

    Two layouts are supported:
    - A simple extracted-frame tree of .jpg/.jpeg/.png files with labels in the path.
    - The official Drive&Act-style tree with activities_3s CSV annotations and MP4 files.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        image_size: tuple[int, int] = (448, 448),
        labels: Sequence[str] = CANONICAL_LABELS,
        camera_view: str = "kinect_color",
        annotation_level: str = "midlevel",
        split: str | None = None,
        frames_per_segment: int = 1,
        activity_mapping: Mapping[str, str] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.image_size = image_size
        self.labels = tuple(labels)
        self.camera_view = camera_view
        self.annotation_level = annotation_level
        self.split = split
        self.frames_per_segment = max(1, frames_per_segment)
        self.activity_mapping = dict(activity_mapping or DRIVEACT_ACTIVITY_TO_LABEL)

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")
        if not self.dataset_root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.dataset_root}")

        self.driveact_root = self._find_driveact_root()
        self.annotation_csv = self._find_annotation_csv() if self.driveact_root else None

    def __iter__(self) -> Iterator[FrameSample]:
        """Yield labeled samples, skipping unknown labels and unreadable media."""
        if self.annotation_csv is not None:
            yield from self._iter_driveact_video_samples()
            return

        for image_path in self._iter_image_paths():
            label = self._infer_label(image_path)
            if label is None:
                LOGGER.warning("Skipping frame with unknown label: %s", image_path)
                continue
            try:
                image = self._load_image(image_path)
            except OSError as exc:
                LOGGER.warning("Skipping unreadable image %s: %s", image_path, exc)
                continue
            yield FrameSample(image=image, label=label, path=image_path)

    def __len__(self) -> int:
        """Return the number of mapped annotation samples or image files."""
        if self.annotation_csv is not None:
            return sum(1 for _ in self._iter_mapped_annotation_rows()) * self.frames_per_segment
        return sum(1 for _ in self._iter_image_paths())

    def _iter_driveact_video_samples(self) -> Iterator[FrameSample]:
        if self.driveact_root is None or self.annotation_csv is None:
            return

        captures: dict[Path, cv2.VideoCapture] = {}
        try:
            for row in self._iter_mapped_annotation_rows():
                label = self._map_activity(row["activity"])
                if label is None:
                    continue

                video_path = self.driveact_root / self.camera_view / f"{row['file_id']}.mp4"
                if not video_path.exists():
                    LOGGER.warning("Skipping annotation with missing video: %s", video_path)
                    continue

                frame_start = int(row["frame_start"])
                frame_end = int(row["frame_end"])
                for frame_index in self._sample_frame_indices(frame_start, frame_end, self.frames_per_segment):
                    image = self._read_video_frame(captures, video_path, frame_index)
                    if image is None:
                        LOGGER.warning("Skipping unreadable frame %s from %s", frame_index, video_path)
                        continue
                    virtual_path = Path(f"{video_path}#frame={frame_index}")
                    yield FrameSample(image=image, label=label, path=virtual_path)
        finally:
            for capture in captures.values():
                capture.release()

    def _iter_mapped_annotation_rows(self) -> Iterator[dict[str, str]]:
        if self.annotation_csv is None:
            return
        with self.annotation_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if self._map_activity(row["activity"]) in self.labels:
                    yield row

    def _find_driveact_root(self) -> Path | None:
        candidates = [
            self.dataset_root,
            self.dataset_root / "Drive&Act",
            self.dataset_root / "Drive&Act" / "Drive&Act",
        ]
        for candidate in candidates:
            if (candidate / "activities_3s").is_dir() and (candidate / self.camera_view).is_dir():
                return candidate
        return None

    def _find_annotation_csv(self) -> Path | None:
        if self.driveact_root is None:
            return None

        filename = f"{self.annotation_level}.chunks_90.csv"
        if self.split is not None:
            filename = f"{self.annotation_level}.chunks_90.{self.split}.csv"

        annotation_csv = self.driveact_root / "activities_3s" / self.camera_view / filename
        if annotation_csv.exists():
            LOGGER.info("Using Drive&Act annotations from %s", annotation_csv)
            return annotation_csv

        LOGGER.warning("Drive&Act annotation CSV not found: %s", annotation_csv)
        return None

    def _iter_image_paths(self) -> Iterator[Path]:
        for path in sorted(self.dataset_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path

    def _infer_label(self, image_path: Path) -> str | None:
        relative_text = image_path.relative_to(self.dataset_root).as_posix().lower()
        for alias, label in LABEL_ALIASES.items():
            if alias in relative_text and label in self.labels:
                return label
        return None

    def _load_image(self, image_path: Path) -> Image.Image:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return image.resize(self.image_size, Image.Resampling.LANCZOS)

    def _map_activity(self, activity: str) -> str | None:
        label = self.activity_mapping.get(activity)
        if label in self.labels:
            return label
        return None

    def _read_video_frame(
        self,
        captures: dict[Path, cv2.VideoCapture],
        video_path: Path,
        frame_index: int,
    ) -> Image.Image | None:
        capture = captures.get(video_path)
        if capture is None:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                return None
            captures[video_path] = capture

        for candidate_index in self._fallback_frame_indices(frame_index):
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate_index)
            ok, frame_bgr = capture.read()
            if ok and frame_bgr is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                return image.resize(self.image_size, Image.Resampling.LANCZOS)
        return None

    @staticmethod
    def _sample_frame_indices(frame_start: int, frame_end: int, count: int) -> tuple[int, ...]:
        frame_start = max(0, frame_start)
        frame_end = max(frame_start + 1, frame_end)
        if count == 1:
            return ((frame_start + frame_end - 1) // 2,)

        span = frame_end - frame_start
        return tuple(frame_start + min(span - 1, round((span - 1) * i / (count - 1))) for i in range(count))

    @staticmethod
    def _fallback_frame_indices(frame_index: int) -> tuple[int, ...]:
        frame_index = max(0, frame_index)
        return (
            frame_index,
            max(0, frame_index - 1),
            frame_index + 1,
            max(0, frame_index - 3),
            frame_index + 3,
        )


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB numpy array for engines that expect arrays."""
    return np.asarray(image.convert("RGB"))


__all__ = [
    "CANONICAL_LABELS",
    "DRIVEACT_ACTIVITY_TO_LABEL",
    "DriveAndActLoader",
    "FrameSample",
    "pil_to_numpy",
]
