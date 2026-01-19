import json
from importlib import resources

from src.vgl.core.base_executor import BaseExecutor
from src.vgl.core.utils.caption_processor import CaptionProcessor
from src.vgl.core.utils.image_processor import ImageProcessor
from src.vgl.gateway.llm_gateway import LLMGateway


class FiboEditExecutor(BaseExecutor):
    default_llm = "vertex_ai/gemini-2.5-flash"

    def __init__(self, llm=None):
        self.llm = llm or self.default_llm
    
    def prepare_prompts_and_schema(self, user_prompt: str, base_path: str):
        package_path = f"src.vgl.config.{base_path}"
        pkg = resources.files(package_path)

        input_schema_text = pkg.joinpath("input_schema.txt").read_text()
        system_prompt = pkg.joinpath("system_prompt.txt").read_text().format_map(
            {"llm_input_schema": input_schema_text}
        )
        final_user_prompt = pkg.joinpath("final_prompt.txt").read_text().format_map(
            {"user_prompt": user_prompt}
        )
        with pkg.joinpath("output_schema.json").open("r") as f:
            output_schema = json.load(f)

        return system_prompt, final_user_prompt, output_schema

    def execute(self, user_prompt: str, seed=None, model_name=None, **kwargs) -> str:
        image = kwargs.get("image")
        mask_image = kwargs.get("mask_image")

        is_mask = mask_image is not None
        base_path = "fibo_edit_mask" if is_mask else "fibo_edit"

        system_prompt, final_prompt, response_schema = self.prepare_prompts_and_schema(
            user_prompt=user_prompt, 
            base_path=base_path
        )

        image = ImageProcessor.apply_gray_mask(image, mask_image) if is_mask else image

        target_model = model_name if model_name is not None else self.llm

        res = LLMGateway.call(
            model_name=target_model,
            system_prompt=system_prompt,
            user_prompt=final_prompt,
            images=[image] if image else [],
            seed=seed,
            response_schema=response_schema
        )

        if isinstance(res, str):
            res = json.loads(res)

        res_with_esthetics_score = CaptionProcessor.add_aesthetics_scores(res)
        clean_caption = CaptionProcessor.prepare_clean_caption(res_with_esthetics_score)

        return clean_caption

