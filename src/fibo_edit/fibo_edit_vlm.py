import gc

import torch
from diffusers.modular_pipelines import ModularPipelineBlocks

# Module-level cache for pipeline
_pipeline_cache = {}


def generate_prompt_local(image, instruction, model="briaai/FIBO-edit-prompt-to-JSON"):
    """Generate edit prompt JSON using local VLM pipeline."""
    global _pipeline_cache

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
