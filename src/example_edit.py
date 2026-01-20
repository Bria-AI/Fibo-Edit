import argparse
from pathlib import Path

import torch
import ujson
from diffusers import BriaFiboEditPipeline
from PIL import Image

from edit_promptify import get_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Fibo Edit example script")
    parser.add_argument(
        "--model-id",
        type=str,
        default="briaai/Fibo-Edit",
        help="Model ID (default: briaai/Fibo-Edit)",
    )
    parser.add_argument(
        "--vlm-mode",
        type=str,
        choices=["api", "local"],
        default="api",
        help="VLM mode: 'api' for cloud-based (Gemini), 'local' for local model",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="gemini/gemini-2.5-flash",
        help="VLM model for prompt generation (default: gemini/gemini-2.5-flash for api mode)",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=["src/example_image.jpg"],
        help="Image path(s) to edit (default: src/example_image.jpg)",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        nargs="+",
        default=["change the car color to green"],
        help="Edit instruction(s) (default: 'change the car color to green')",
    )
    parser.add_argument(
        "--masks",
        type=str,
        nargs="+",
        default=None,
        help="Mask image path(s) for inpainting (white=edit region). Can use 'none' for no mask.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Guidance scale (default: 5.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path("generated")
    output_dir.mkdir(exist_ok=True)

    # Load pipeline once
    pipeline = BriaFiboEditPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )
    pipeline.to("cuda")

    # Normalize masks list
    masks = args.masks or [None] * len(args.images)
    masks = [None if m and m.lower() == "none" else m for m in masks]

    # Build list of (image_path, instruction, mask_path) tuples
    if len(args.instructions) == 1 and len(masks) == 1:
        # Single instruction and mask applied to all images
        tuples = [(img, args.instructions[0], masks[0]) for img in args.images]
    elif len(args.images) == 1 and len(masks) == 1:
        # Single image and mask with multiple instructions
        tuples = [(args.images[0], instr, masks[0]) for instr in args.instructions]
    elif len(args.images) == len(args.instructions) == len(masks):
        # Matched tuples
        tuples = list(zip(args.images, args.instructions, masks))
    elif len(args.images) == len(args.instructions) and len(masks) == 1:
        # Matched image/instruction pairs with single mask
        tuples = [(img, instr, masks[0]) for img, instr in zip(args.images, args.instructions)]
    else:
        raise ValueError(
            f"Mismatch: {len(args.images)} images, {len(args.instructions)} instructions, "
            f"and {len(masks)} masks. Provide matching counts or use 1 for broadcast."
        )

    # Validate local VLM mode does not support masks
    if args.vlm_mode == "local":
        if any(m is not None for m in masks):
            raise ValueError("Local VLM mode does not support masks. Use --vlm-mode api for masked editing.")

    # Process each tuple
    for image_path, instruction, mask_path in tuples:
        image = Image.open(image_path)
        img_stem = Path(image_path).stem

        # Load mask if provided (as grayscale)
        mask = None
        if mask_path:
            mask = Image.open(mask_path).convert("L")

        print(f"Processing: {image_path}")
        print(f"Instruction: {instruction}")
        if mask_path:
            print(f"Mask: {mask_path}")

        prompt_json_str = get_prompt(
            image=image,
            instruction=instruction,
            mask_image=mask,
            model=args.vlm_model,
            vlm_mode=args.vlm_mode,
        )

        pipeline_kwargs = {
            "prompt": prompt_json_str,
            "image": image,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
        }
        if mask is not None:
            pipeline_kwargs["mask"] = mask

        output = pipeline(**pipeline_kwargs).images[0]

        # Save outputs
        output_image_path = output_dir / f"{img_stem}_edited.jpg"
        output_text_path = output_dir / f"{img_stem}.txt"

        output.save(output_image_path)
        with open(output_text_path, "w") as f:
            ujson.dump(prompt_json_str, f, escape_forward_slashes=False)

        print(f"Saved: {output_image_path}, {output_text_path}\n")


if __name__ == "__main__":
    main()
