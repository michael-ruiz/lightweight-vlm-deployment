"""Fine-tune Qwen2.5-VL-3B on Drive&Act dataset using LoRA."""

import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def main() -> None:
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    output_dir = "models/qwen2.5-vl-3b-driveact-lora"
    dataset_path = "data/finetune/dataset.jsonl"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run create_finetune_dataset.py first.")

    LOGGER.info("Loading processor for %s", model_id)
    # We restrict the vision encoder to 280x280 max_pixels.
    # The image is divided into patches, 280x280 is small enough to massively speed up inference
    # while still allowing the model to distinguish driver actions after fine-tuning.
    max_pixels = 280 * 280
    processor = AutoProcessor.from_pretrained(
        model_id, 
        max_pixels=max_pixels,
    )
    
    LOGGER.info("Loading dataset")
    # Load dataset using Hugging Face datasets library
    dataset = load_dataset("json", data_files={"train": dataset_path})["train"]
    
    # 4-bit Quantization Config to fit in 24GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    LOGGER.info("Loading model in 4-bit")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA Config
    # We target the attention projections which is standard for LLM/VLM fine-tuning
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Formatting function for SFTTrainer
    def format_data(example):
        # We need to process the conversation format into model inputs
        # But SFTTrainer with recent transformers can handle the conversation list directly
        # if the tokenizer has a chat_template.
        # So we just return the dataset as is, the DataCollator / SFTTrainer handles it 
        # using the processor.apply_chat_template under the hood if configured correctly,
        # or we manually format.
        # Actually, the easiest way with TRL is to use a custom collator or map the dataset.
        text = processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        
        # We need to extract the image path and load it
        image_path = example["messages"][0]["content"][0]["image"]
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        # Return dict with text and images
        return {"text": text, "images": [image]}
        
    LOGGER.info("Formatting dataset")
    # TRL SFTTrainer handles formatting differently in recent versions.
    # The recommended approach for VLMs is formatting it ahead of time into `text` and `images` columns,
    # then providing a custom DataCollator. Or using the default SFTTrainer if it supports it.
    
    # For robust Qwen2.5-VL training, we can use the dataset directly and rely on processor.
    # Let's map the dataset to standard format.
    def preprocess_function(examples):
        texts = []
        images_list = []
        for msgs in examples["messages"]:
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            
            # The image path is in the first message's first content item
            img_path = msgs[0]["content"][0]["image"]
            from PIL import Image
            images_list.append([Image.open(img_path).convert("RGB")])
            
        batch = processor(
            text=texts,
            images=images_list,
            padding=True,
            return_tensors="pt"
        )
        # SFTTrainer needs labels, for causal LM labels = input_ids
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # SFTConfig
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=2048, # Enough for image tokens + prompt
        dataset_text_field="text",
        remove_unused_columns=False,
    )
    
    # TRL's SFTTrainer with data collator
    # We will use the format_data map and rely on the processor's default DataCollator for VLMs
    dataset = dataset.map(format_data)
    
    def data_collator(features):
        texts = [f["text"] for f in features]
        images = [f["images"] for f in features]
        # flatten images
        flat_images = [img for sublist in images for img in sublist]
        
        batch = processor(
            text=texts,
            images=flat_images,
            padding=True,
            return_tensors="pt"
        )
        batch["labels"] = batch["input_ids"].clone()
        
        # Mask out padding in labels so it doesn't affect loss
        batch["labels"][batch["attention_mask"] == 0] = -100
        return batch

    LOGGER.info("Starting training")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )
    
    trainer.train()
    
    LOGGER.info("Saving LoRA adapters to %s", output_dir)
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
