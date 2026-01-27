from vgl.gateway.llm_gateway import LLMGateway


class BaseExecutor():
    def __init__(self, llm=None):
        self.llm = "vertex_ai/gemini-2.5-flash" if llm is None else llm

    def execute(self, model_name, system_prompt, user_prompt, 
                images, temperature, seed, response_schema, 
                max_output_tokens, top_p, **kwargs) -> str:
        
        res  = LLMGateway.call(model_name=model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        images=images,
                        temperature=temperature,
                        seed=seed,
                        response_schema=response_schema,
                        max_output_tokens=max_output_tokens,
                        top_p=top_p)
        
        return res