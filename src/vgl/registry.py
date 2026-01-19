from enum import Enum
from typing import NamedTuple, Type

from .core.executors.fibo_edit_executor import FiboEditExecutor
from .models.edit_model import ImageAnalysisEdit


class PromptMode(Enum):
    EDIT = "edit"

class ExecutorConfig(NamedTuple):
    executor: Type
    model_object: Type

EXECUTOR_REGISTRY = {
    PromptMode.EDIT: ExecutorConfig(
        executor=FiboEditExecutor, 
        model_object=ImageAnalysisEdit
    )
}

