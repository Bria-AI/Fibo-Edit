from typing import List, Optional

from pydantic import BaseModel, Field


class ObjectDescription(BaseModel):
    description: str = Field(..., description="Short description of the object.")
    location: str = Field(..., description="E.g., 'center', 'top-left', 'bottom-right foreground'.")
    relationship: str = Field(..., description="Describe the relationship between the object and the other objects in the image.")
    relative_size: Optional[str] = Field(None, description="E.g., 'small', 'medium', 'large within frame'.")
    shape_and_color: Optional[str] = Field(None, description="Describe the basic shape and dominant color.")
    texture: Optional[str] = Field(None, description="E.g., 'smooth', 'rough', 'metallic', 'furry'.")
    appearance_details: Optional[str] = Field(None, description="Any other notable visual details.")
    # If cluster of object
    number_of_objects: Optional[int] = Field(None, description="The number of objects in the cluster.")
    # Human-specific fields
    pose: Optional[str] = Field(None, description="Describe the body position.")
    expression: Optional[str] = Field(None, description="Describe facial expression.")
    clothing: Optional[str] = Field(None, description="Describe attire.")
    action: Optional[str] = Field(None, description="Describe the action of the human.")
    gender: Optional[str] = Field(None, description="Describe the gender of the human.")
    skin_tone_and_texture: Optional[str] = Field(None, description="Describe the skin tone and texture.")
    orientation: Optional[str] = Field(None, description="Describe the orientation of the human.")

class LightingDetails(BaseModel):
    conditions: str = Field(..., description="E.g., 'bright daylight', 'dim indoor', 'studio lighting', 'golden hour'.")
    direction: str = Field(..., description="E.g., 'front-lit', 'backlit', 'side-lit from left'.")
    shadows: Optional[str] = Field(None, description="Describe the presence of shadows.")

class AestheticsDetails(BaseModel):
    composition: str = Field(..., description="E.g., 'rule of thirds', 'symmetrical', 'centered', 'leading lines'.")
    color_scheme: str = Field(..., description="E.g., 'monochromatic blue', 'warm complementary colors', 'high contrast'.")
    mood_atmosphere: str = Field(..., description="E.g., 'serene', 'energetic', 'mysterious', 'joyful'.")
    aesthetic_score: str = Field(default_factory=lambda: "very high", description="E.g., 'very low', 'low', 'medium', 'high', 'very high'.")
    preference_score: str = Field(default_factory=lambda: "very high", description="E.g., 'very low', 'low', 'medium', 'high', 'very high'.")

class PhotographicCharacteristicsDetails(BaseModel):
    depth_of_field: str = Field(..., description="E.g., 'shallow', 'deep', 'bokeh background'.")
    focus: str = Field(..., description="E.g., 'sharp focus on subject', 'soft focus', 'motion blur'.")
    camera_angle: str = Field(..., description="E.g., 'eye-level', 'low angle', 'high angle', 'dutch angle'.")
    lens_focal_length: str = Field(..., description="E.g., 'wide-angle', 'telephoto', 'macro', 'fisheye'.")

class TextRender(BaseModel):
    text: str = Field(..., description="The text content.")
    location: str = Field(..., description="E.g., 'center', 'top-left', 'bottom-right foreground'.")
    size: str = Field(..., description="E.g., 'small', 'medium', 'large within frame'.")
    color: str = Field(..., description="E.g., 'red', 'blue', 'green'.")
    font: str = Field(..., description="E.g., 'realistic', 'cartoonish', 'minimalist'.")
    appearance_details: Optional[str] = Field(None, description="Any other notable visual details.")

class ImageAnalysis(BaseModel):
    short_description: str = Field(..., description="A concise summary of the image content, 200 words maximum.")
    objects: List[ObjectDescription] = Field(..., description="List of prominent foreground/midground objects.")
    background_setting: str = Field(..., description="Describe the overall environment, setting, or background, including any notable background elements.")
    lighting: LightingDetails = Field(..., description="Details about the lighting.")
    aesthetics: AestheticsDetails = Field(..., description="Details about the image aesthetics.")
    photographic_characteristics: Optional[PhotographicCharacteristicsDetails] = Field(None, description="Details about photographic characteristics.")
    style_medium: Optional[str] = Field(None, description="Identify the artistic style or medium.")
    text_render: Optional[List[TextRender]] = Field(None, description="List of text renders in the image.")
    context: str = Field(..., description="Provide any additional context that helps understand the image better.")
    artistic_style: Optional[str] = Field(None, description="describe specific artistic characteristics, 3 words maximum.")