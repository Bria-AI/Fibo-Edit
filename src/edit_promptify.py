import os

from src.vgl import generate_prompt
from src.vgl.registry import PromptMode


def get_prompt(image, instruction, mask_image=None, model="gemini/gemini-2.5-flash", vlm_mode="api"):
    if vlm_mode == "local":
        from src.fibo_edit_vlm import generate_prompt_local
        return generate_prompt_local(image=image, instruction=instruction, model=model)
    else:
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