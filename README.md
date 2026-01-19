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

<p align="center">
  <b>Fibo Edit is the new edition to the Fibo Family - a set of models that follow the VGL (Visual GenAI Language) Paradigm.</b><br>
  <b>This 8B parameter image-to-image diffusion model is designed for precise, deterministic, and controllable editing.</b>
  <b></b>
  <br><br>
</p>

<h2>🌍 What's Fibo Edit?</h2>
<p>Unlike traditional edit models that rely on loose text prompts, <b>Fibo Edit</b> introduces a new paradigm of structured control. It operates on <b>Structured JSON inputs</b> paired with a source image and a mask, enabling explicit, interpretable, and repeatable editing workflows optimized for professional production environments.</p>

<p>Developed by Bria AI, Fibo Edit prioritizes transparency, legal safety, and granular control, ranking among the <b>top 3 editing models</b> in open benchmarks for its aesthetic quality and adherence.</p>

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
  <li><a href="https://docs.bria.ai/image-generation/v2-endpoints/image-edit">Bria.ai</a></li>
  <li><a href="https://fal.ai/models/bria/fibo-edit">Fal.ai</a></li>
  <li><a href="https://replicate.com/bria/fibo-edit">Replicate</a></li>
  <li><a href="https://platform.bria.ai/labs/fibo-edit">Bria Fibo Lab</li>
</ul>

<p><b>Source-Code & Weights</b></p>

<ul>
  <li>The model is open source for non-commercial use with <a href="https://creativecommons.org/licenses/by-nc/4.0/deed.en">this license</a>.</li>
  <li>For commercial use <a href="https://bria.ai/contact-us">Click here</a>.</li>
</ul>
    
<h2>Quick Start Guide</h2>
<p> clone the repository and install the requirements</p>
<pre><code class="language-bash">git clone https://github.com/briaai/Fibo-Edit.git
cd Fibo-Edit
</code></pre>
<p>install the requirements</p>
<pre><code class="language-bash">pip install git+https://github.com/huggingface/diffusers torch torchvision openai boltons ujson sentencepiece accelerate transformers
</code></pre>

<h3>Promptify Setup</h3>
<p>The <code>edit_promptify</code> module uses a VLM to analyze images and generate structured JSON prompts. By default, it uses <a href="https://openrouter.ai">OpenRouter</a> to access the underlying VLM.</p>

<p><b>Option 1: OpenRouter API (Recommended)</b></p>
<ol>
  <li>Create an account at <a href="https://openrouter.ai">openrouter.ai</a></li>
  <li>Generate an API key from your dashboard</li>
  <li>Set the API key as an environment variable:
    <pre><code class="language-bash">export OPENROUTER_API_KEY=your_openrouter_api_key</code></pre>
  </li>
</ol>

<p><b>Option 2: Use Your Own LLM (Advanced)</b></p>
<p>For advanced users who prefer to use a different LLM provider, you can use the schemas and system prompts directly with any compatible model. The structured prompt schemas and system prompts are available in the <code>src/</code> directory and can be adapted for use with OpenAI, Anthropic, local models, or any other LLM of your choice.</p>

<h3>Image + Mask</h3>

***EXAMPLE IMAGE HERE***

```python
import torch
from diffusers import DiffusionPipeline
from PIL import Image
from edit_promptify import edit_image_with_mask

# 1. Load the pipeline
pipe = DiffusionPipeline.from_pretrained(
    "briaai/Fibo-Edit",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# 2. Load your source image and mask
source_image = Image.open("path/to/image.png")
mask_image = Image.open("path/to/mask.png")

# 3. Generate structured JSON prompt using edit_promptify
# This uses a VLM to analyze the image and create a detailed structured prompt
prompt = edit_image_with_mask(source_image, "change the car to red velvet texture", mask_image)

# 4. Run the edit
result = pipe(
    image=source_image,
    mask_image=mask_image,
    prompt=prompt,
    num_inference_steps=50
).images[0]

result.save("fibo_edit_result.png")
```


<h3>Only Image</h3>

***EXAMPLE IMAGE HERE***

```python
import torch
from diffusers import DiffusionPipeline
from PIL import Image
from edit_promptify import edit_image

# 1. Load the pipeline
pipe = DiffusionPipeline.from_pretrained(
    "briaai/Fibo-Edit",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# 2. Load your source image
source_image = Image.open("path/to/image.png")

# 3. Generate structured JSON prompt using edit_promptify
# This uses a VLM to analyze the image and create a detailed structured prompt
prompt = edit_image(source_image, "change car to a motorcycle")

# 4. Run the edit
result = pipe(
    image=source_image,
    prompt=prompt,
    num_inference_steps=50
).images[0]

result.save("fibo_edit_result.png")
```

# Need to replace the examples with the new ones
## more Examples
<div class="image-row">
  <figure>
    <img src="https://bria-public.s3.us-east-1.amazonaws.com/original.png" alt="original image"/>
    <figcaption>original image</figcaption>
  </figure>
  <figure>
    <img src="https://bria-public.s3.us-east-1.amazonaws.com/no_prompt.png" alt="No prompt"/>
    <figcaption>Inspire #1: No prompt</figcaption>
  </figure>
  <figure>
    <img src="https://bria-public.s3.us-east-1.amazonaws.com/make_futuristic.png" alt="Make futuristic"/>
    <figcaption>Inspire #2: Make futuristic</figcaption>
  </figure>
  
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
<p>see the examples in the <a href="examples">examples</a> directory for more details.</p>
<h2>🧠 Training and Architecture</h2>
<p><strong>Fibo Edit</strong> is an 8B-parameter DiT-based, flow-matching text-to-image model trained <strong>exclusively on licensed data</strong> and on <strong>&gt; long, structured JSON captions</strong> (~1,000 words each), enabling strong prompt adherence and professional-grade control. It uses <strong>SmolLM3-3B</strong> as the text encoder with a novel <strong>DimFusion</strong> conditioning architecture for efficient long-caption training, and <strong>Wan 2.2</strong> as the VAE. The structured supervision promotes native disentanglement for targeted, iterative refinement without prompt drift, while VLM-assisted prompting expands short user intents, fills in missing details, and extracts/edits structured prompts from images using our fine-tuned <strong>Qwen-2.5</strong>-based VLM or <strong>Gemini 2.5 Flash</strong>. For reproducibility, we provide the assistant system prompt and the structured-prompt JSON schema across the “Generate,” “Refine,” and “Inspire” modes.</p>


# Need to replace the examples with the new ones

<h2 id="More Samples">More Samples</h2>
<p>Generate</p>
<img src="https://bria-public.s3.us-east-1.amazonaws.com/Generate.png" alt="Benchmark Chart" width="800"/>
<p>Inspire & Refine</p>
<img src="https://bria-public.s3.us-east-1.amazonaws.com/Refine.ong.png" alt="Benchmark Chart" width="800"/>

  
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
