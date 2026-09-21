from __future__ import annotations

import json

from yaj.models import Question, QuestionType

SYSTEM_PROMPT = (
    "You are a precise classifier. Given the provided context, answer the "
    "question exactly as instructed. Do not explain your reasoning."
)


def option_labels(n: int) -> list[str]:
    """Return single-character labels A..T for up to 20 options."""
    return [chr(ord("A") + i) for i in range(n)]


def _serialize_state(state: str | dict | list) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def _format_noul(q: Question) -> str:
    criteria = q.criteria
    assert isinstance(criteria, dict)
    true_crit = criteria.get("true", criteria.get("True", ""))
    false_crit = criteria.get("false", criteria.get("False", ""))
    return (
        f"Question: {q.instructions}\n\n"
        f"Criteria for True: {true_crit}\n"
        f"Criteria for False: {false_crit}\n\n"
        "Answer with exactly one word: Yes or No"
    )


def _format_choice(q: Question) -> str:
    criteria = q.criteria
    assert isinstance(criteria, dict)
    keys = list(criteria.keys())
    labels = option_labels(len(keys))
    options_lines = [
        f"{label}. {key}: {criteria[key]}" for label, key in zip(labels, keys)
    ]
    options_text = "\n".join(options_lines)
    return (
        f"Question: {q.instructions}\n\n"
        f"Options:\n{options_text}\n\n"
        "Answer with exactly one letter (A, B, ...)"
    )


def _format_score(q: Question) -> str:
    criteria = q.criteria
    assert isinstance(criteria, list)
    levels_lines = [f"{i}: {level}" for i, level in enumerate(criteria)]
    levels_text = "\n".join(levels_lines)
    return (
        f"Question: {q.instructions}\n\n"
        f"Levels:\n{levels_text}\n\n"
        "Answer with exactly one digit (0, 1, ...)"
    )


_FORMATTERS = {
    QuestionType.NOUL: _format_noul,
    QuestionType.CHOICE: _format_choice,
    QuestionType.SCORE: _format_score,
}


def build_messages(
    state: str | dict | list, question: Question
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _serialize_state(state)},
        {"role": "user", "content": _FORMATTERS[question.type](question)},
    ]
