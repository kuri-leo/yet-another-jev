from __future__ import annotations

import enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class QuestionType(str, enum.Enum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


class Question(BaseModel):
    type: QuestionType
    instructions: str
    criteria: dict[str, str] | list[str]


class DecisionRequest(BaseModel):
    model: str
    state: str | dict | list
    questions: dict[str, Question]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: dict[str, float] | None = None
    confidence: float | None = None


Answer = Annotated[
    Union[NoulAnswer, ChoiceAnswer, ScoreAnswer],
    Field(discriminator="type"),
]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class DecisionResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)
