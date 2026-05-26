"""Create Qwen2.5-VL SFT dataset from Drive&Act segments."""

import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from src.data_loader import DriveAndActLoader
from src.evaluator import DRIVEACT_QWEN_PROMPT

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def main() -> None:
    dataset_root = Path("data")
    output_dir = dataset_root / "finetune"
    image_dir = output_dir / "images"
    jsonl_path = output_dir / "dataset.jsonl"
    
    image_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("Loading Drive&Act dataset from %s", dataset_root)
    loader = DriveAndActLoader(dataset_root, image_size=(532, 532))
    
    # We want to create 1-frame examples for fine-tuning since our goal is to 
    # regain accuracy with single-frame (or minimal frames) at lower resolution.
    frames_per_segment = 1
    segments = loader.get_segment_stream(
        frames_per_segment=frames_per_segment,
        batch_size=1,
    )
    
    dataset_records: list[dict[str, Any]] = []
    
    # Process up to a certain limit or all segments.
    # For a real fine-tuning run, you'd process the whole training set.
    # DriveAndActLoader currently yields everything it finds.
    count = 0
    for batch in tqdm(segments, desc="Processing segments"):
        for segment in batch:
            # Save the middle frame as a JPEG
            if not segment.frames:
                continue
                
            frame = segment.frames[0]
            # frame is a numpy array (H, W, 3)
            img = Image.fromarray(frame)
            
            img_filename = f"segment_{count:05d}_{segment.label.replace(' ', '_')}.jpg"
            img_path = image_dir / img_filename
            img.save(img_path, format="JPEG", quality=90)
            
            # Construct the conversational format expected by Qwen2.5-VL and SFTTrainer
            # The image path should be absolute or relative to the training script.
            # Using absolute path for safety during training.
            abs_img_path = img_path.absolute().as_posix()
            
            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": abs_img_path},
                            {"type": "text", "text": DRIVEACT_QWEN_PROMPT}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": segment.label}
                        ]
                    }
                ]
            }
            dataset_records.append(record)
            count += 1

    # Write to JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in dataset_records:
            f.write(json.dumps(record) + "\n")
            
    LOGGER.info("Successfully created %d training examples at %s", len(dataset_records), jsonl_path)

if __name__ == "__main__":
    main()
