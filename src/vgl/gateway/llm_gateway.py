import base64
import io
from typing import List, Optional

import litellm
from PIL import Image

# litellm._turn_on_debug()

class LLMGateway:
    @staticmethod
    def _prepare_image_base64(image: Image.Image) -> str:
        image = image.convert("RGB")
        max_pixels = 262144 
        if image.size[0] * image.size[1] > max_pixels:
            ratio = (max_pixels / (image.size[0] * image.size[1]))**0.5
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @classmethod
    def call(cls, 
             model_name: str, 
             system_prompt: str, 
             user_prompt: str, 
             images: Optional[List[Image.Image]] = None,
             temperature: float = 0.0, 
             seed: Optional[int] = 42,
             response_schema: Optional[dict] = None,
             max_output_tokens: int = 5000,
             top_p: float = 1.0,
             **kwargs):
        
        user_content = [{"type": "text", "text": user_prompt}]
        
        if images:
            for img in images:
                b64_data = cls._prepare_image_base64(img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
                })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # 1. Handle Response Format
        # OpenAI/Gemini support "json_object" but some models require specific schema placement
        response_format = None
        if response_schema:
            response_format = {
                "type": "json_object",
                "response_schema": response_schema 
            }

        # 2. Extract Provider-Specific Params
        # Check if the model is Vertex AI or Gemini to apply specific logic
        is_gemini = "gemini" in model_name.lower() or "vertex_ai" in model_name.lower()
        
        extra_body = kwargs.get("extra_body", {})
        metadata = kwargs.get("metadata", {})

        if is_gemini:
            # Only add thinking_config for Gemini models
            if "thinking_config" not in extra_body:
                extra_body["thinking_config"] = {"include_thoughts": True, "thinking_budget": 0}
            # Add billing category metadata for Gemini
            metadata.setdefault("billing_category", "production")

        response = litellm.completion(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_output_tokens,
            # seed=seed,
            response_format=response_format,
            extra_body=extra_body,
            metadata=metadata,
        )
        
        if response.choices[0].finish_reason == 'length':
            raise Exception(f"Max tokens ({max_output_tokens}) reached for model {model_name}")
    
        return response.choices[0].message.content