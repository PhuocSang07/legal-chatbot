# Data model
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Các Documents có liên qua đến câu hỏi của người dùng không , 'yes' hoặc 'no'"
    )