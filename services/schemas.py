from pydantic import BaseModel, field_validator
from typing import List, Optional


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Pergunta não pode ser vazia")
        if len(v) > 2000:
            raise ValueError("Pergunta excede 2000 caracteres")
        return v
