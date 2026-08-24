import json
from importlib import resources

from PIL import Image

from vgl.core.base_executor import BaseExecutor
from vgl.core.utils.caption_processor import CaptionProcessor
from vgl.core.utils.image_processor import ImageProcessor
from vgl.gateway.llm_gateway import LLMGateway


class FiboEditExecutor(BaseExecutor):
    default_llm = "vertex_ai/gemini-2.5-flash"

    def __init__(self, llm=None):
        self.llm = llm or self.default_llm
    
    # Schema files a mode does not carry are read from here: the multi-reference mode uses
    # the same input and output schema as the single-reference mode.
    shared_schema_path = "fibo_edit"

    def prepare_prompts_and_schema(self, user_prompt: str, base_path: str):
        package_path = f"vgl.config.{base_path}"
        pkg = resources.files(package_path)
        shared = resources.files(f"vgl.config.{self.shared_schema_path}")

        def schema_file(name):
            own = pkg.joinpath(name)
            return own if own.is_file() else shared.joinpath(name)

        input_schema_text = schema_file("input_schema.txt").read_text()
        system_prompt = pkg.joinpath("system_prompt.txt").read_text().format_map(
            {"llm_input_schema": input_schema_text}
        )
        final_user_prompt = pkg.joinpath("final_prompt.txt").read_text().format_map(
            {"user_prompt": user_prompt}
        )
        with schema_file("output_schema.json").open("r") as f:
            output_schema = json.load(f)

        return system_prompt, final_user_prompt, output_schema

    def execute(self, user_prompt: str, seed=None, model_name=None, **kwargs) -> str:
        image = kwargs.get("image")
        mask_image = kwargs.get("mask_image")

        if image is None:
            raise ValueError("At least one reference image is required.")

        images = list(image) if isinstance(image, (list, tuple)) else [image]
        if not images:
            raise ValueError("At least one reference image is required.")
        if any(not isinstance(reference, Image.Image) for reference in images):
            raise ValueError("Every reference image must be a PIL image.")

        is_mask = mask_image is not None
        if is_mask and len(images) > 1:
            raise ValueError("Mask-based editing supports a single reference image only.")
        if len(images) > 1:
            base_path = "fibo_edit_multiref"
        elif is_mask:
            base_path = "fibo_edit_mask"
        else:
            base_path = "fibo_edit"

        system_prompt, final_prompt, response_schema = self.prepare_prompts_and_schema(
            user_prompt=user_prompt, 
            base_path=base_path
        )

        if is_mask:
            images[0] = ImageProcessor.apply_gray_mask(images[0], mask_image)

        target_model = model_name if model_name is not None else self.llm

        res = LLMGateway.call(
            model_name=target_model,
            system_prompt=system_prompt,
            user_prompt=final_prompt,
            images=images,
            seed=seed,
            response_schema=response_schema
        )

        if isinstance(res, str):
            res = json.loads(res)

        res_with_esthetics_score = CaptionProcessor.add_aesthetics_scores(res)
        clean_caption = CaptionProcessor.prepare_clean_caption(res_with_esthetics_score)

        return clean_caption
