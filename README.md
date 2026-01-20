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
<p> clone the repository and install the requirements</p>
<pre><code class="language-bash">git clone https://github.com/briaai/Fibo-Edit.git
cd Fibo-Edit
</code></pre>
<p>install the requirements</p>
<pre><code class="language-bash">pip install git+https://github.com/huggingface/diffusers torch torchvision openai boltons ujson sentencepiece accelerate transformers
</code></pre>

<h3>Promptify Setup</h3>
<p>The repository supports two modes for generating structured JSON prompts:</p>

<p><b>API Mode (default):</b> Uses Gemini as the VLM. Set your API key with <code class="language-bash">export GEMINI_API_KEY="your-api-key"</code></p>

<p><b>Local Mode:</b> Uses a local VLM model (<code>briaai/FIBO-edit-prompt-to-JSON</code>) via diffusers ModularPipelineBlocks. No API key required, runs entirely on your GPU.</p>

```bash
# API mode (default)
python src/example_edit.py --images photo.jpg --instructions "change the car color to green"

# Local mode
python src/example_edit.py --vlm-mode local --vlm-model briaai/FIBO-edit-prompt-to-JSON --images photo.jpg --instructions "change the car color to green"
```

<p><b>Note:</b> Local VLM mode does not support mask-based editing. Use API mode (<code>--vlm-mode api</code>) for masked edits.</p>

<h3>Image + Mask</h3>

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

<img src="https://bria-public.s3.us-east-1.amazonaws.com/Edit+Assets/RemoveObjects.png" alt="onlyImage" width="800"/>

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
<p>see the examples in the <a href="examples">examples</a> directory for more details.</p>


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
