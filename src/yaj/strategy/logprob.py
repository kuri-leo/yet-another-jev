from __future__ import annotations

import logging
import math

from yaj.client import LLMClient
from yaj.models import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    QuestionType,
    ScoreAnswer,
)
from yaj.prompt import option_labels
from yaj.strategy.base import Strategy

logger = logging.getLogger(__name__)


class UnsupportedError(Exception):
    """Raised when logprob strategy cannot handle the question."""


class LogprobExtractionError(Exception):
    """Raised when expected tokens are missing from logprobs."""


def _softmax(logprobs: dict[str, float]) -> dict[str, float]:
    max_lp = max(logprobs.values())
    exps = {k: math.exp(v - max_lp) for k, v in logprobs.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def _confidence(probs: dict[str, float]) -> float:
    n = len(probs)
    if n <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
    max_entropy = math.log(n)
    if max_entropy <= 0:
        return 1.0
    conf = 1.0 - (entropy / max_entropy)
    return max(0.0, min(1.0, conf))


_YES_TOKENS = {"Yes", "yes", "Y", "YES", "True", "true"}
_NO_TOKENS = {"No", "no", "N", "NO", "False", "false"}


class LogprobStrategy(Strategy):
    async def decide(
        self, client: LLMClient | None, messages: list[dict], question: Question
    ) -> Answer:
        if question.type == QuestionType.CHOICE:
            assert isinstance(question.criteria, dict)
            if len(question.criteria) > 20:
                raise UnsupportedError(
                    f"Logprob strategy supports ≤20 choice options, got {len(question.criteria)}"
                )
        elif question.type == QuestionType.SCORE:
            assert isinstance(question.criteria, list)
            if len(question.criteria) > 10:
                raise UnsupportedError(
                    f"Logprob strategy supports ≤10 score levels, got {len(question.criteria)}"
                )

        assert client is not None
        response = await client.complete(
            messages,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            temperature=0,
        )

        choice = response.choices[0]
        if not choice.logprobs or not choice.logprobs.content:
            raise LogprobExtractionError(
                "LLM response contains no logprobs — the model or provider may not support logprobs"
            )
        top_logprobs_list = choice.logprobs.content[0].top_logprobs
        if not top_logprobs_list:
            raise LogprobExtractionError(
                "LLM response has empty top_logprobs list"
            )
        raw = {entry.token: entry.logprob for entry in top_logprobs_list}
        logger.debug(
            "logprob tokens for %s question: %s",
            question.type.value,
            {t: round(lp, 3) for t, lp in sorted(raw.items(), key=lambda x: -x[1])[:10]},
        )

        if question.type == QuestionType.NOUL:
            return self._handle_noul(raw)
        elif question.type == QuestionType.CHOICE:
            return self._handle_choice(raw, question)
        else:
            return self._handle_score(raw, question)

    def _handle_noul(self, raw: dict[str, float]) -> NoulAnswer:
        yes_lps = [
            lp
            for t, lp in raw.items()
            if t in _YES_TOKENS or t.strip().lower() in {"yes", "y", "true"}
        ]
        no_lps = [
            lp
            for t, lp in raw.items()
            if t in _NO_TOKENS or t.strip().lower() in {"no", "n", "false"}
        ]
        yes_lp = max(yes_lps, default=None)
        no_lp = max(no_lps, default=None)
        if yes_lp is None and no_lp is None:
            raise LogprobExtractionError(
                f"No Yes/No tokens found in top_logprobs. "
                f"Got tokens: {list(raw.keys())}"
            )
        # If only one side found, treat missing side as extremely unlikely
        if yes_lp is None:
            yes_lp = -100.0
        if no_lp is None:
            no_lp = -100.0
        probs = _softmax({"yes": yes_lp, "no": no_lp})
        return NoulAnswer(noul=probs["yes"])

    def _handle_choice(
        self, raw: dict[str, float], question: Question
    ) -> ChoiceAnswer:
        assert isinstance(question.criteria, dict)
        keys = list(question.criteria.keys())
        labels = option_labels(len(keys))
        label_lps = {}
        for label in labels:
            lp = raw.get(label, raw.get(f" {label}"))
            if lp is None:
                lp = raw.get(label.lower(), raw.get(f" {label.lower()}"))
            if lp is not None:
                label_lps[label] = lp
        if not label_lps:
            raise LogprobExtractionError(
                f"No option tokens ({labels}) found in top_logprobs. "
                f"Got tokens: {list(raw.keys())}"
            )
        probs = _softmax(label_lps)
        # Map back to original keys (only for found labels)
        found_keys = [k for k, l in zip(keys, labels) if l in label_lps]
        key_probs = {k: probs[l] for k, l in zip(keys, labels) if l in label_lps}
        # Normalize key_probs to sum to 1
        total = sum(key_probs.values())
        key_probs = {k: v / total for k, v in key_probs.items()}
        best_key = max(key_probs, key=lambda k: key_probs[k])
        return ChoiceAnswer(
            choice=best_key,
            probabilities=key_probs,
            confidence=_confidence(key_probs),
        )

    def _handle_score(
        self, raw: dict[str, float], question: Question
    ) -> ScoreAnswer:
        assert isinstance(question.criteria, list)
        n_levels = len(question.criteria)
        digits = [str(i) for i in range(n_levels)]
        digit_lps = {}
        for d in digits:
            lp = raw.get(d, raw.get(f" {d}"))
            if lp is not None:
                digit_lps[d] = lp
        if not digit_lps:
            raise LogprobExtractionError(
                f"No score tokens ({digits}) found in top_logprobs. "
                f"Got tokens: {list(raw.keys())}"
            )
        probs = _softmax(digit_lps)
        # Normalize over found digits only
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        expected = sum(int(d) * p for d, p in probs.items())
        legend = {str(i): label for i, label in enumerate(question.criteria)}
        return ScoreAnswer(
            score=expected,
            legend=legend,
            probabilities=probs,
            confidence=_confidence(probs),
        )
