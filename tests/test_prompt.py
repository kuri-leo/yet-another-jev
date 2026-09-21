from __future__ import annotations

import json

from yaj.models import Question, QuestionType
from yaj.prompt import SYSTEM_PROMPT, build_messages, option_labels


def test_system_prompt_exact_wording():
    expected = (
        "You are a precise classifier. Given the provided context, "
        "answer the question exactly as instructed. Do not explain your reasoning."
    )
    assert SYSTEM_PROMPT == expected


def test_build_messages_structure():
    q = Question(
        type=QuestionType.NOUL,
        instructions="Is this urgent?",
        criteria={"true": "Threatening to leave", "false": "Routine"},
    )
    msgs = build_messages("Customer says: I quit", q)
    assert len(msgs) == 3
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "user"


def test_noul_prompt_format():
    q = Question(
        type=QuestionType.NOUL,
        instructions="Is this urgent?",
        criteria={"true": "Threatening to leave", "false": "Routine"},
    )
    msgs = build_messages("Customer says: I quit", q)
    content = msgs[2]["content"]
    expected = (
        "Question: Is this urgent?\n\n"
        "Criteria for True: Threatening to leave\n"
        "Criteria for False: Routine\n\n"
        "Answer with exactly one word: Yes or No"
    )
    assert content == expected
    assert "Yes" in content and "No" in content


def test_choice_prompt_format():
    q = Question(
        type=QuestionType.CHOICE,
        instructions="Which department?",
        criteria={"billing": "Payments", "technical": "Bugs", "sales": "Pricing"},
    )
    msgs = build_messages("Invoice issue", q)
    content = msgs[2]["content"]
    expected = (
        "Question: Which department?\n\n"
        "Options:\n"
        "A. billing: Payments\n"
        "B. technical: Bugs\n"
        "C. sales: Pricing\n\n"
        "Answer with exactly one letter (A, B, ...)"
    )
    assert content == expected
    assert "A." in content and "B." in content and "C." in content


def test_score_prompt_format():
    q = Question(
        type=QuestionType.SCORE,
        instructions="Rate urgency",
        criteria=["Low", "Normal", "High"],
    )
    msgs = build_messages("Help!", q)
    content = msgs[2]["content"]
    expected = (
        "Question: Rate urgency\n\n"
        "Levels:\n"
        "0: Low\n"
        "1: Normal\n"
        "2: High\n\n"
        "Answer with exactly one digit (0, 1, ...)"
    )
    assert content == expected
    assert "0:" in content and "1:" in content and "2:" in content


def test_option_labels():
    assert option_labels(0) == []
    assert option_labels(3) == ["A", "B", "C"]
    labels_20 = option_labels(20)
    assert len(labels_20) == 20
    assert labels_20[0] == "A"
    assert labels_20[19] == "T"


def test_state_string_passed_directly():
    q = Question(
        type=QuestionType.NOUL,
        instructions="test",
        criteria={"true": "yes", "false": "no"},
    )
    msgs = build_messages("raw text context", q)
    assert msgs[1]["content"] == "raw text context"


def test_state_dict_serialized_as_json():
    q = Question(
        type=QuestionType.NOUL,
        instructions="test",
        criteria={"true": "yes", "false": "no"},
    )
    data = {"key": "value", "nested": {"count": 1}}
    msgs = build_messages(data, q)
    assert json.loads(msgs[1]["content"]) == data


def test_state_list_serialized_as_json():
    q = Question(
        type=QuestionType.NOUL,
        instructions="test",
        criteria={"true": "yes", "false": "no"},
    )
    items = [{"id": 1}, {"id": 2}]
    msgs = build_messages(items, q)
    assert json.loads(msgs[1]["content"]) == items


def test_state_unicode_preserved():
    q = Question(
        type=QuestionType.NOUL,
        instructions="test",
        criteria={"true": "yes", "false": "no"},
    )
    data = {"text": "你好世界", "emoji": "🚀"}
    msgs = build_messages(data, q)
    assert "你好世界" in msgs[1]["content"]
    assert "🚀" in msgs[1]["content"]
