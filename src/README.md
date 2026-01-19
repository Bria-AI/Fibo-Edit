# Edit Promptify

A Python module for generating structured JSON descriptions for **Fibo Edit** image editing tasks.

This module uses vision-capable AI models to analyze images and generate the structured JSON format required by the Fibo Edit diffusion model.

## Features

- **Edit Mode**: Transform images based on text prompts
- **Pydantic Validation**: Ensures well-structured, validated JSON output
- **LiteLLM Integration**: Supports multiple LLM providers

## Installation

From the project root:

```bash
uv sync
```

## Setup

Set your Gemini API key:

```bash
export GEMINI_API_KEY="your-api-key"
```

Get an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Quick Start

```python
from PIL import Image
from src.edit_promptify import get_prompt

image = Image.open("photo.jpg")
result = get_prompt(image, "make the sky sunset colors")
print(result)
```

### Using Different Models

```python
from src.edit_promptify import get_prompt

result = get_prompt(
    image=image,
    instruction="add a cat on the couch",
    model="gemini/gemini-2.5-pro"
)
```

## API Reference

### `get_prompt(image, instruction, model="gemini/gemini-2.5-flash")`

Generate a structured JSON prompt for image editing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `PIL.Image` | required | Input image |
| `instruction` | `str` | required | Edit instruction (e.g., "make the dog golden") |
| `model` | `str` | `gemini/gemini-2.5-flash` | LiteLLM model identifier |

**Returns**: JSON string with structured image description.

## Supported Models

Any vision-capable model supported by [LiteLLM](https://docs.litellm.ai/docs/providers):

| Model | Identifier | Required Env Var |
|-------|------------|------------------|
| Gemini 2.5 Flash (default) | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| Gemini 2.5 Pro | `gemini/gemini-2.5-pro` | `GEMINI_API_KEY` |

## Output JSON Schema

The module generates JSON with the following structure:

```json
{
  "short_description": "Concise summary of the image (max 200 words)",
  "objects": [
    {
      "description": "Detailed object description",
      "location": "Position in frame (e.g., 'center', 'top-left')",
      "relationship": "Relationship to other objects",
      "relative_size": "small | medium | large",
      "shape_and_color": "Basic shape and dominant color",
      "texture": "Surface quality",
      "appearance_details": "Other visual details",
      "pose": "For humans: body position",
      "expression": "For humans: facial expression",
      "clothing": "For humans: attire description",
      "action": "For humans: current action",
      "gender": "For humans: apparent gender",
      "skin_tone_and_texture": "For humans: skin details",
      "orientation": "Positioning (e.g., 'facing left')",
      "number_of_objects": "For clusters: count"
    }
  ],
  "background_setting": "Environment description",
  "lighting": {
    "conditions": "Lighting type",
    "direction": "Light source direction",
    "shadows": "Shadow characteristics"
  },
  "aesthetics": {
    "composition": "Compositional style",
    "color_scheme": "Color palette",
    "mood_atmosphere": "Overall mood",
    "preference_score": "very low | low | medium | high | very high",
    "aesthetic_score": "very low | low | medium | high | very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "DOF description",
    "focus": "Focus point",
    "camera_angle": "Camera position",
    "lens_focal_length": "Lens type"
  },
  "style_medium": "Artistic medium (e.g., 'photograph', 'oil painting')",
  "artistic_style": "Style characteristics (max 3 words)",
  "context": "General image type description",
  "text_render": [
    {
      "text": "Text content",
      "location": "Position",
      "size": "Text size",
      "color": "Text color",
      "font": "Font style"
    }
  ],
  "edit_instruction": "Imperative command for the edit"
}
```

## Files

| File | Description |
|------|-------------|
| `edit_promptify.py` | Core module with prompt generation |
| `example_edit.py` | Usage example |
| `example_image.jpg` | Sample image for testing |

## Command Line Interface

The `example_edit.py` script provides a CLI for running edits:

```bash
export GEMINI_API_KEY="your-key"
python src/example_edit.py
```

### CLI Options

```
--model MODEL           LLM model for prompt generation (default: gemini/gemini-2.5-flash)
--images PATH [PATH...] Image path(s) to edit (default: src/example_image.jpg)
--instructions TEXT [TEXT...]
                        Edit instruction(s) (default: 'change the car color to green')
--num-inference-steps N Number of inference steps (default: 50)
--guidance-scale SCALE  Guidance scale (default: 5.0)
```

### Examples

```bash
# Default (uses example_image.jpg)
python src/example_edit.py

# Single image with custom instruction
python src/example_edit.py --images photo.jpg --instructions "make it vintage"

# Multiple images with one instruction
python src/example_edit.py --images a.jpg b.jpg --instructions "add sunset lighting"

# One image with multiple instructions
python src/example_edit.py --images photo.jpg --instructions "make vintage" "add rain"

# Custom model and parameters
python src/example_edit.py --images photo.jpg --instructions "add snow" \
    --model gemini/gemini-2.5-pro --num-inference-steps 30 --guidance-scale 7.0
```

Outputs are saved to `generated/<image_stem>_edited.jpg` and `generated/<image_stem>.txt`.

## Integration with Fibo Edit

This module generates the structured JSON that Fibo Edit expects. Use the output as the `prompt` parameter:

```python
import torch
from diffusers import BriaFiboEditPipeline
from PIL import Image
from src.edit_promptify import get_prompt

# Generate structured JSON
image = Image.open("photo.jpg")
edit_json = get_prompt(image, "make it look vintage")

# Use with Fibo Edit
pipe = BriaFiboEditPipeline.from_pretrained(
    "briaai/Fibo-Edit",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

result = pipe(
    image=image,
    prompt=edit_json,
    num_inference_steps=50,
    guidance_scale=5
).images[0]

result.save("edited.png")
```

## License

See the main repository LICENSE file for details.
