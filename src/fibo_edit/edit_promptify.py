import os

from fibo_edit.fibo_edit_vlm import generate_prompt_local
from vgl import generate_prompt
from vgl.registry import PromptMode


def get_prompt(image, instruction, mask_image=None, model="gemini/gemini-2.5-flash", vlm_mode="api"):
    """Generate edit prompt using either API or local VLM mode.

    Args:
        image: PIL Image to edit, or an ordered list of PIL reference images
        instruction: Edit instruction string
        mask_image: Optional mask image (API mode only, single reference only)
        model: Model identifier
        vlm_mode: "api" for cloud-based (Gemini), "local" for local model
    """
    references = list(image) if isinstance(image, (list, tuple)) else [] if image is None else [image]
    if not references:
        raise ValueError("At least one reference image is required.")
    if mask_image is not None and len(references) > 1:
        raise ValueError("Mask-based editing supports a single reference image only.")

    if vlm_mode == "local":
        if len(references) > 1:
            raise ValueError(
                "Local VLM mode supports one reference image only. Use vlm_mode=\"api\", which supports multiple references."
            )
        if mask_image is not None:
            raise ValueError("Local VLM mode does not support masked edits. Use vlm_mode=\"api\" for masked edits.")
        return generate_prompt_local(image, instruction, mask_image=mask_image, model=model)

    # API mode
    if model.startswith("gemini"):
        assert os.environ.get("GEMINI_API_KEY"), "GEMINI_API_KEY is not set"
    result_json_str = generate_prompt(
        mode=PromptMode.EDIT,
        llm=model,
        user_prompt=instruction,
        image=image,
        mask_image=mask_image,
    )
    return result_json_str
