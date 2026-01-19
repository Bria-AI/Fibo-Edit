class ModelAdapter:
    """Handles provider-specific logic to keep the Gateway agnostic."""
    
    @staticmethod
    def get_provider_params(model_name: str, **kwargs) -> dict:
        """
        Maps model names to their specific configuration requirements.
        """
        model_lower = model_name.lower()
        
        # Detect if we are using Gemini (AI Studio or Vertex AI)
        is_gemini = "gemini" in model_lower or "vertex_ai" in model_lower
        
        # Start with generic containers
        extra_body = kwargs.get("extra_body", {})
        metadata = kwargs.get("metadata", {})
        allowed_params = None

        if is_gemini:
            # Maintain exact thinking_config logic for Gemini
            if "thinking_config" not in extra_body:
                extra_body["thinking_config"] = {
                    "include_thoughts": True, 
                    "thinking_budget": 0
                }
            
            # Maintain exact billing_category metadata
            metadata.setdefault("billing_category", "production")
            
            # Whitelist seed for Vertex AI to prevent LiteLLM from stripping it
            if "vertex_ai" in model_lower:
                allowed_params = ["seed"]

        return {
            "extra_body": extra_body if extra_body else None,
            "metadata": metadata if metadata else None,
            "allowed_openai_params": allowed_params
        }