import os

from vgl import generate_prompt
from vgl.registry import PromptMode

from fibo_edit.fibo_edit_vlm import generate_prompt_local


def get_prompt(image, instruction, mask_image=None, model="gemini/gemini-2.5-flash", vlm_mode="api"):
    """Generate edit prompt using either API or local VLM mode.

    Args:
        image: PIL Image to edit
        instruction: Edit instruction string
        mask_image: Optional mask image (not supported in local mode)
        model: Model identifier
        vlm_mode: "api" for cloud-based (Gemini), "local" for local model
    """
    if vlm_mode == "local":
        return generate_prompt_local(image, instruction, model=model)

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