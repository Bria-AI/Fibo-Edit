from typing import Optional

from pydantic import Field

from .base_model import ImageAnalysis


class ImageAnalysisEdit(ImageAnalysis):
    edit_instruction: Optional[str] = Field(
        None,
        description="A concise but clear description of the expected changes in the image.",
    )
