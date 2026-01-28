<p align="center">
  <img src="https://bria-public.s3.us-east-1.amazonaws.com/Bria-logo.svg" width="200"/>
</p>

 <a href="https://huggingface.co/briaai/Fibo-Edit" target="_blank">
    <img
      alt="Model Card"
      src="https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?logo=huggingface&logoColor=black&style=for-the-badge"
    />
  </a>
  &nbsp;


  <a href="https://huggingface.co/spaces/briaai/Fibo-Edit" target="_blank">
    <img
      alt="Hugging Face Demo"
      src="https://img.shields.io/badge/Hugging%20Face-Demo-FFD21E?logo=huggingface&logoColor=black&style=for-the-badge"
    />
  </a>
  &nbsp;

  <a href="https://platform.bria.ai" target="_blank">
    <img
      alt="Bria Platform"
      src="https://img.shields.io/badge/Bria-Platform-0EA5E9?style=for-the-badge"
    />
  </a>
  &nbsp;

  <a href="https://discord.com/invite/Nxe9YW9zHS" target="_blank">
    <img
      alt="Bria Discord"
      src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white&style=for-the-badge"
    />
  </a>
</p>

<p align="center">
  <img src="https://bria-public.s3.us-east-1.amazonaws.com/Edit+Assets/RecolorHero.jpeg" width="1024" alt="Fibo Edit Hero Image"/>
</p>
<p align="center">
  <b>FIBO-Edit brings the power of structured prompt generation to image editing.</b><br>
  Built on Fibo's</a> foundation and of JSON-native control, FIBO-Edit delivers precise, deterministic, and fully controllable edits. No ambiguity, no surprises.
  <b></b>
  <br><br>
</p>

<h2>🌍 What's Fibo Edit?</h2>
<p>Most image editing models rely on loose, ambiguous text prompts, but not FIBO-Edit. FIBO-Edit introduces a new paradigm of structured control, operating on structured JSON inputs paired with a source image (and optionally a mask). This enables explicit, interpretable, and repeatable editing workflows optimized for professional production environments.</p>

<p>Developed by Bria AI, FIBO-Edit prioritizes transparency, legal safety, and granular control: ranking among the top models in open benchmarks for prompt adherence and quality.</p>


<p>📄 <i>Technical report coming soon.</i> For architecture details, see <a href="https://huggingface.co/briaai/FIBO">FIBO</a>.</p>

<h2>📐 The VGL Paradigm</h2>
<p>FIBO-Edit is natively built on <a href="https://docs.bria.ai/vgl">Visual GenAI Language (VGL)</a>. VGL standardizes image generation by replacing vague natural language descriptions with explicit, human-machine-readable JSON. By disentangling visual elements—such as lighting, composition, style, and camera parameters—VGL transforms editing from a probabilistic guessing game into a deterministic engineering task. Fibo-Edit reads these structured blueprints to perform precise updates without prompt drift, ensuring the output matches your exact specifications.</p>

<h2> News</h2>
<ul>
  <li>2026-1-16: Fibo Edit released on Hugging Face 🎉</li>
  <li>2026-1-16: Integrated with Diffusers library 🧨</li>
</ul>


<h2>🔑 Key Features</h2>
<ul>
  <li><b>Structured JSON Control</b>: Move beyond "prompt drift." Define edits with explicit parameters (lighting, composition, style) using a structured JSON format for deterministic results.</li>
  <li><b>Native Masking</b>: Built-in support for mask-based editing allows you to target specific regions of an image with pixel-perfect precision, leaving the rest untouched.</li>
  <li><b>Production-Ready Architecture</b>: At 8B parameters, the model balances high-fidelity output with the speed and efficiency required for commercial pipelines.</li>
  <li><b>Deep Customization</b>: The lightweight architecture empowers researchers to build specialized "Edit" models for domain-specific tasks without compromising quality.</li>
  <li><b>Responsible & Licensed</b>: Trained exclusively on fully licensed data, ensuring zero copyright infringement risks for commercial users.</li>
</ul>

<h2>⚡ Quick Start</h2>

<p align="center">
  🚀 <a href="https://huggingface.co/spaces/briaai/Fibo-Edit">Try Fibo Edit now →</a>
</p>

<p>Fibo Edit is available everywhere you build, either as source-code and weights, ComfyUI nodes or API endpoints.</p>

<p><b>API Endpoint:</b></p>
<ul>
  <li><a href="https://docs.bria.ai/image-editing/v2-endpoints/edit-image">Bria.ai</a></li>
  <li><a href="https://fal.ai/models/bria/fibo-edit/edit">Fal.ai</a></li>
  <li><a href="https://replicate.com/bria/fibo-edit">Replicate (Coming soon)</a></li>
  <li><a href="https://platform.bria.ai/labs/fibo-edit">Bria Fibo Lab</a></li>
</ul>

<p><b>Source-Code & Weights</b></p>
<ul>
  <li>The model is open source for non-commercial use with <a href="https://creativecommons.org/licenses/by-nc/4.0/deed.en">this license</a> </li>
  <li>For commercial use <a href="https://bria.ai/contact-us?hsCtaAttrib=114250296256">Click here</a>.</li>
</ul>

<h2>Quick Start Guide</h2>
<p>Clone the repository and install dependencies:</p>
<pre><code class="language-bash">git clone https://github.com/Bria-AI/Fibo-Edit.git
cd Fibo-Edit
uv sync
</code></pre>

<h3>Promptify Setup</h3>
<p>The repository supports two modes for generating structured JSON prompts:</p>

<p><b>API Mode (default):</b> Uses Gemini as the VLM. Set your API key with <code class="language-bash">export GEMINI_API_KEY="your-api-key"</code></p>

<p><b>Local Mode:</b> Uses a local VLM model (<code>briaai/FIBO-edit-prompt-to-JSON</code>) via diffusers ModularPipelineBlocks. No API key required, runs entirely on your GPU.</p>

```bash
# API mode (default)
uv run python scripts/example_edit.py --images photo.jpg --instructions "change the car color to green"

# Local mode
uv run python scripts/example_edit.py --vlm-mode local --vlm-model briaai/FIBO-edit-prompt-to-JSON --images photo.jpg --instructions "change the car color to green"
```

<p><b>Note:</b> Local VLM mode does not support mask-based editing. Use API mode (<code>--vlm-mode api</code>) for masked edits.</p>

<h3>Image + Mask</h3>

```python
import torch
from diffusers import BriaFiboEditPipeline
from PIL import Image

from fibo_edit.edit_promptify import get_prompt

# 1. Load the pipeline
pipeline = BriaFiboEditPipeline.from_pretrained(
        "briaai/Fibo-Edit",
        torch_dtype=torch.bfloat16,
    )
pipeline.to("cuda")

# 2. Load your source image and mask
source_image = Image.open("examples/example_image.jpg")
mask_image = Image.open("examples/example_mask.jpg")

# 3. Generate structured JSON prompt using edit_promptify
# This uses a VLM to analyze the image and create a detailed structured prompt
prompt = get_prompt(image=source_image, instruction="change the car color to green", mask_image=mask_image)
# 4. Run the edit
result = pipeline(
    image=source_image,
    mask=mask_image,
    prompt=prompt,
    num_inference_steps=50
).images[0]

result.save("fibo_edit_result.png")
```


<h3>Only Image</h3>

<img src="https://bria-public.s3.us-east-1.amazonaws.com/Edit+Assets/RemoveObjects.png" alt="onlyImage" width="800"/>

```python
import torch
from diffusers import BriaFiboEditPipeline
from PIL import Image

from fibo_edit.edit_promptify import get_prompt

# 1. Load the pipeline
pipeline = BriaFiboEditPipeline.from_pretrained(
        "briaai/Fibo-Edit",
        torch_dtype=torch.bfloat16,
    )
pipeline.to("cuda")

# 2. Load your source image and mask
source_image = Image.open("examples/example_image.jpg")

# 3. Generate structured JSON prompt using edit_promptify
# This uses a VLM to analyze the image and create a detailed structured prompt
prompt = get_prompt(image=source_image, instruction="change the car color to green")

# 4. Run the edit
result = pipeline(
    image=source_image,
    prompt=prompt,
    num_inference_steps=50
).images[0]

result.save("fibo_edit_result.png")
```
<table>
  <tr>
    <td align="center">
      <img src="docs/assets/Relight.gif" width="400" alt="Relight"/>
    </td>
    <td align="center">
      <img src="docs/assets/Restyle.gif" width="400" alt="Restyle"/>
    </td>
  </tr>
</table>
<table>
  <tr>
    <td align="center">
      <img src="docs/assets/Retype.gif" width="400" alt="Retype"/>
    </td>
    <td align="center">
      <img src="docs/assets/Recolor.gif" width="400" alt="Recolor"/>
    </td>
  </tr>
</table>
</div>
<h3>Advanced Usage</h3>
<details>
  <summary>Gemini Setup [optional]</summary>
  <p>FIBO supports any VLM as part of the pipeline. To use Gemini as VLM backbone for FIBO, follow these instructions:</p>
  <ol>
    <li>
      <p><b>Obtain a Gemini API Key</b><br/>
      Sign up for the <a href="https://aistudio.google.com/app/apikey">Google AI Studio (Gemini)</a> and create an API key.</p>
    </li>
    <li>
      <p><b>Set the API Key as an Environment Variable</b><br/>
      Store your Gemini API key in the <code>GEMINI_API_KEY</code> environment variable:</p>
      <pre><code class="language-bash">export GEMINI_API_KEY=your_gemini_api_key
</code></pre>
      <p>You can add the above line to your <code>.bashrc</code>, <code>.zshrc</code>, or similar shell profile for persistence.</p>
    </li>
  </ol>
</details>

<h3>Running Example Scripts</h3>
<p>As an alternative to the Python snippets above, you can use the provided example script:</p>

```bash
uv run python scripts/example_edit.py --images examples/example_image.jpg --instructions "change the car color to green"
```

<h4>More Examples</h4>

```bash
# Multiple images with one instruction
uv run python scripts/example_edit.py --images a.jpg b.jpg --instructions "add sunset lighting"

# One image with multiple instructions
uv run python scripts/example_edit.py --images photo.jpg --instructions "make vintage" "add rain"

# Custom model and parameters
uv run python scripts/example_edit.py --images photo.jpg --instructions "add snow" \
    --model gemini/gemini-2.5-pro --num-inference-steps 30 --guidance-scale 7.0

# With a LoRA model
uv run python scripts/example_edit.py --images photo.jpg --instructions "turn this image into an impressionist oil painting" --lora /path/to/lora
```

<h4>CLI Options</h4>

```
--model                  LLM model for prompt generation (default: gemini/gemini-2.5-flash)
--images                 Image path(s) to edit
--instructions           Edit instruction(s)
--num-inference-steps    Number of inference steps (default: 50)
--guidance-scale         Guidance scale (default: 5.0)
--lora                   Path to LoRA checkpoint
--lora-scale             LoRA weight scale (default: 1.0)
```

<h2>Finetuning</h2>
<p>Fibo-Edit supports LoRA finetuning to adapt the model to your specific editing tasks and domains.</p>

<h3>Dataset Preparation</h3>
<p>Prepare a directory with paired input/output images and a <code>metadata.csv</code> file:</p>

```
dataset/
├── input_image1.jpg      # Source image (before edit)
├── output_image1.jpg     # Target image (after edit)
├── input_image2.jpg
├── output_image2.jpg
└── metadata.csv
```

<p>The <code>metadata.csv</code> must have three columns:</p>

```csv
input_file_name,output_file_name,caption
input_image1.jpg,output_image1.jpg,"{""short_description"":""A red car"",""edit_instruction"":""Change color to red""}"
input_image2.jpg,output_image2.jpg,"{""mood"":""warm"",""edit_instruction"":""Add sunset lighting""}"
```

<h3>Caption Format</h3>
<p>Captions must be valid JSON strings. The <code>edit_instruction</code> key is recommended to describe the edit operation. You can include other VGL fields as needed:</p>

<details>
  <summary>Full JSON Schema</summary>

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
</details>

<h3>Training Command</h3>

```bash
uv run python scripts/finetune_fibo_edit.py \
  --instance_data_dir /path/to/dataset \
  --output_dir /path/to/output \
  --lora_rank 64 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 1000 \
  --checkpointing_steps 250 \
  --learning_rate 1e-4 \
  --gradient_checkpointing 1
```

<h4>Key Arguments</h4>

```
--instance_data_dir            Dataset directory containing metadata.csv
--output_dir                   Directory to save checkpoints
--lora_rank                    LoRA rank, 64 recommended for most use cases (default: 128)
--max_train_steps              Total training steps, 1000-2000 recommended (default: 1501)
--checkpointing_steps          Save checkpoint every N steps (default: 250)
--gradient_checkpointing       Enable gradient checkpointing to reduce VRAM (default: 1)
--train_batch_size             Batch size per device (default: 1)
--gradient_accumulation_steps  Gradient accumulation steps (default: 4)
--learning_rate                Learning rate (default: 1e-4)
--resume_from_checkpoint       Path to checkpoint or "latest" to resume training
```

<p>See <code>scripts/finetune_fibo_edit.py --help</code> for all available options.</p>

<h3>Using the Finetuned Model</h3>
<p>Use <code>scripts/example_edit.py</code> with the <code>--lora</code> flag to load your finetuned checkpoint:</p>

```bash
uv run python scripts/example_edit.py \
  --images input.jpg \
  --instructions "your edit instruction" \
  --lora /path/to/output/checkpoint_1000 \
  --lora-scale 1.0
```

<p>Or in Python:</p>

```python
from diffusers import BriaFiboEditPipeline
import torch

pipeline = BriaFiboEditPipeline.from_pretrained("briaai/Fibo-Edit", torch_dtype=torch.bfloat16)
pipeline.to("cuda")

# Load and fuse LoRA weights
pipeline.load_lora_weights("/path/to/output/checkpoint_1000")
pipeline.fuse_lora(lora_scale=1.0)

# Use the pipeline as normal
result = pipeline(image=source_image, prompt=prompt, num_inference_steps=50).images[0]
```

<h3>Tips</h3>
<ul>
  <li>Start with <code>--lora_rank 64</code> for most use cases; increase to 128 for more complex adaptations</li>
  <li>Enable <code>--gradient_checkpointing 1</code> to reduce VRAM usage (enabled by default)</li>
  <li>Checkpoints are saved as <code>checkpoint_250/</code>, <code>checkpoint_500/</code>, etc.</li>
  <li>Use <code>--train_batch_size 1</code> when training on variable resolution images</li>
  <li>For multi-GPU training, use <code>accelerate launch</code> with appropriate configuration</li>
</ul>

## Get Involved

<p>If you have questions about this repository, feedback to share, or want to contribute directly, we welcome your issues and pull requests on GitHub. Your contributions help make FIBO better for everyone.</p>
<p>If you're passionate about fundamental research, we're hiring full-time employees (FTEs) and research interns. Don't wait - reach out to us at hr@bria.ai</p>

## Citation
We kindly encourage citation of our work if you find it useful.

```bibtex
@article{gutflaish2025generating,
  title={Generating an Image From 1,000 Words: Enhancing Text-to-Image With Structured Captions},
  author={Gutflaish, Eyal and Kachlon, Eliran and Zisman, Hezi and Hacham, Tal and Sarid, Nimrod and Visheratin, Alexander and Huberman, Saar and Davidi, Gal and Bukchin, Guy and Goldberg, Kfir and others},
  journal={arXiv preprint arXiv:2511.06876},
  year={2025}
}
```
<p align="center"><b>❤️ FIBO model card and ⭐ Star FIBO on GitHub to join the movement for responsible generative AI!</b></p>
