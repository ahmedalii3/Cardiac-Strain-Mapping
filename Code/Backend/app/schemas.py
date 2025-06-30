from pydantic import BaseModel

class LocalizationResult(BaseModel):
    cropped_images: list
    center_of_mass: list
    median: list
