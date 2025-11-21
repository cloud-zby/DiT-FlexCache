from pydantic import BaseModel, Field
from .utils import generate_task_id
from pydantic import field_validator
from typing import Literal


class TaskRequest(BaseModel):
    task_id: str = Field(
        default_factory=generate_task_id, description="Task ID (auto-generated)"
    )
    prompt: str = Field("", description="Generation prompt")
    negative_prompt: str = Field("", description="Negative prompt")
    first_image_path: str = Field("", description="Base64 encoded image or URL")
    last_image_path: str = Field("", description="Base64 encoded image or URL")
    save_video_path: str = Field(
        "", description="Save video path (optional, defaults to task_id.mp4)"
    )
    target_video_length: int = Field(81, description="Target video length")
    target_video_resolution: str = Field(
        "720p", description="such as 720p, 1080p, 480p"
    )
    seed: int = Field(42, description="Random seed")
    aspect_ratio: Literal["16:9", "9:16", "4:3", "3:4", "1:1", "2:3", "3:2"] = Field(
        "16:9", description="Allowed values: 16:9, 9:16, 4:3, 3:4, 1:1, 2:3, 3:2"
    )

    @field_validator("aspect_ratio")
    def validate_aspect_ratio(cls, v):
        allowed = ["16:9", "9:16", "4:3", "3:4", "1:1", "2:3", "3:2"]
        if v not in allowed:
            raise ValueError(
                f"Invalid aspect ratio. Allowed values are: {', '.join(allowed)}"
            )
        return v

    def __init__(self, **data):
        super().__init__(**data)
        if not self.save_video_path:
            self.save_video_path = f"{self.task_id}.mp4"

    def get(self, key, default=None):
        return getattr(self, key, default)


class TaskStatusMessage(BaseModel):
    task_id: str = Field(..., description="Task ID")


class TaskResponse(BaseModel):
    task_id: str
    task_status: str
    save_video_path: str


class StopTaskResponse(BaseModel):
    stop_status: str
    reason: str
