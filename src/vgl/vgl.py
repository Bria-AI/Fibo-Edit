from .registry import EXECUTOR_REGISTRY, PromptMode


def generate_prompt(
    mode: PromptMode,
    user_prompt: str,
    llm: str = None, 
    seed: int = None,
    **kwargs
):    

    config = EXECUTOR_REGISTRY.get(mode)
    executor = config.executor()

    result = executor.execute(
        user_prompt=user_prompt, 
        model_name=llm, 
        seed=seed,
        **kwargs
    )

    return result