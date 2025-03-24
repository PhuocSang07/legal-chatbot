from pydantic import BaseModel, Field
from typing import Optional

class DocumentModel(BaseModel):
    key: Optional[str] = Field(None)
    page_content: Optional[str] = Field(None)
    metadata: dict = Field(default_factory=dict)