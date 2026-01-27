"""
Fuse LoRA weights into the base model and save the result.

Usage:
    python scripts/fuse_lora.py \
        --lora_ckpt_path example_finetune_results/checkpoint_250 \
        --output_path fused_model

Then use the fused model with example_edit.py:
    python scripts/example_edit.py --model-id fused_model --images ...
"""

import argparse
from pathlib import Path

import torch
from diffusers import BriaFiboEditPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse LoRA weights into base model")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="briaai/Fibo-Edit",
        help="Path to pretrained model or model identifier from Hugging Face",
    )
    parser.add_argument(
        "--lora_ckpt_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the fused model",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="Scale factor for LoRA weights (default: 1.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading base model: {args.pretrained_model_name_or_path}")
    pipe = BriaFiboEditPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA weights from: {args.lora_ckpt_path}")
    pipe.load_lora_weights(args.lora_ckpt_path)

    print(f"Fusing LoRA weights with scale={args.lora_scale}")
    pipe.fuse_lora(lora_scale=args.lora_scale)
    pipe.unload_lora_weights()

    print(f"Saving fused model to: {args.output_path}")
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(args.output_path)

    print("Done! Use the fused model with:")
    print(f"  python src/example_edit.py --model-id {args.output_path} --images ...")


if __name__ == "__main__":
    main()
