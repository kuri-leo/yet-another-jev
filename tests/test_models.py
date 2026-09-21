import pytest
from yaj.models import (
    Answer,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    Question,
    QuestionType,
    ScoreAnswer,
    Usage,
)


def test_noul_question_parses():
    q = Question(
        type=QuestionType.NOUL,
        instructions="Is this urgent?",
        criteria={"true": "Threatening to leave", "false": "Routine request"},
    )
    assert q.type == "noul"


def test_choice_question_parses():
    q = Question(
        type=QuestionType.CHOICE,
        instructions="Which department?",
        criteria={"billing": "Payments", "technical": "Bugs"},
    )
    assert q.type == "choice"


def test_score_question_parses():
    q = Question(
        type=QuestionType.SCORE,
        instructions="Rate urgency",
        criteria=["Low", "Normal", "High"],
    )
    assert q.type == "score"
    assert isinstance(q.criteria, list)


def test_decision_request_with_string_state():
    req = DecisionRequest(
        model="test-model",
        state="some text",
        questions={"q1": Question(
            type=QuestionType.NOUL,
            instructions="test",
            criteria={"true": "yes", "false": "no"},
        )},
    )
    assert req.state == "some text"


def test_decision_request_with_dict_state():
    req = DecisionRequest(
        model="test-model",
        state={"key": "value"},
        questions={},
    )
    assert req.state == {"key": "value"}


def test_decision_request_with_list_state():
    req = DecisionRequest(
        model="test-model",
        state=["message 1", "message 2"],
        questions={},
    )
    assert req.state == ["message 1", "message 2"]


def test_noul_answer():
    ans = NoulAnswer(noul=0.85)
    assert ans.type == "noul"
    assert ans.noul == 0.85


def test_choice_answer():
    ans = ChoiceAnswer(
        choice="billing",
        probabilities={"billing": 0.9, "technical": 0.1},
        confidence=0.9,
    )
    assert ans.type == "choice"
    assert ans.choice == "billing"
    assert ans.probabilities == {"billing": 0.9, "technical": 0.1}
    assert ans.confidence == 0.9

    ans_minimal = ChoiceAnswer(choice="billing")
    assert ans_minimal.probabilities is None
    assert ans_minimal.confidence is None


def test_score_answer():
    ans = ScoreAnswer(
        score=2.5,
        probabilities={"1": 0.1, "2": 0.3, "3": 0.6},
        confidence=0.8,
    )
    assert ans.type == "score"
    assert ans.score == 2.5
    assert ans.probabilities == {"1": 0.1, "2": 0.3, "3": 0.6}
    assert ans.confidence == 0.8

    ans_minimal = ScoreAnswer(score=1.0)
    assert ans_minimal.probabilities is None
    assert ans_minimal.confidence is None


def test_decision_response_discriminated_union():
    resp_data = {
        "model": "gpt-4o",
        "answers": {
            "is_urgent": {"type": "noul", "noul": 0.95},
            "category": {"type": "choice", "choice": "billing"},
            "score": {"type": "score", "score": 4.0},
        },
        "usage": {"input_tokens": 120, "output_tokens": 15},
    }
    resp = DecisionResponse.model_validate(resp_data)
    assert isinstance(resp.answers["is_urgent"], NoulAnswer)
    assert isinstance(resp.answers["category"], ChoiceAnswer)
    assert isinstance(resp.answers["score"], ScoreAnswer)
    assert resp.usage.input_tokens == 120
    assert resp.usage.output_tokens == 15


def test_decision_response_default_usage():
    resp = DecisionResponse(
        model="gpt-4o",
        answers={},
    )
    assert resp.usage.input_tokens == 0
    assert resp.usage.output_tokens == 0


def test_decision_response_custom_usage():
    resp = DecisionResponse(
        model="gpt-4o",
        answers={"q1": NoulAnswer(noul=0.5)},
        usage=Usage(input_tokens=10, output_tokens=20),
    )
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 20


def test_decision_response_json_roundtrip():
    original = DecisionResponse(
        model="gpt-4o-mini",
        answers={
            "q_noul": NoulAnswer(noul=0.12),
            "q_choice": ChoiceAnswer(choice="a", probabilities={"a": 1.0}, confidence=1.0),
            "q_score": ScoreAnswer(score=3.5, probabilities={"3": 0.5, "4": 0.5}, confidence=0.7),
        },
        usage=Usage(input_tokens=50, output_tokens=5),
    )
    json_str = original.model_dump_json()
    reconstructed = DecisionResponse.model_validate_json(json_str)
    assert reconstructed == original

