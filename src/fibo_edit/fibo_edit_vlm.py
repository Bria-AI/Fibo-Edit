import gc

import torch
from diffusers.modular_pipelines import ModularPipelineBlocks

# Module-level cache for pipeline
_pipeline_cache = {}


def generate_prompt_local(image, instruction, mask_image=None, model="briaai/FIBO-edit-prompt-to-JSON"):
    """Generate edit prompt JSON locally for one reference image; masks are not supported."""
    if isinstance(image, (list, tuple)):
        if len(image) > 1:
            raise ValueError(
                "Local VLM mode supports one reference image only. Use vlm_mode=\"api\", which supports multiple references."
            )
        if image:
            image = image[0]
    if mask_image is not None:
        raise ValueError("Local VLM mode does not support masked edits. Use vlm_mode=\"api\" for masked edits.")

    # Cache pipeline for reuse
    if model not in _pipeline_cache:
        pipeline = ModularPipelineBlocks.from_pretrained(model, trust_remote_code=True)
        _pipeline_cache[model] = pipeline.init_pipeline()

    pipeline = _pipeline_cache[model]

    # Move to CUDA for inference
    pipeline.to("cuda")

    output = pipeline(image=image, prompt=instruction)

    # Extract the JSON string from PipelineState
    json_prompt = output.values["json_prompt"]

    # Move VLM to CPU to free GPU memory for the main pipeline
    pipeline.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    return json_prompt
