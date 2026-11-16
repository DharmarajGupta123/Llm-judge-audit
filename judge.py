"""Pairwise judge + single-answer rubric scorer.

The pairwise judge is the main mode: given a question, its context, and
two answers, it returns which answer is better as JSON. The rubric
scorer is a second mode: it scores one answer 1-5 on groundedness,
relevance, and completeness.

Both parsers are defensive: model output is expected to be JSON, but
fences, surrounding prose, and lowercase winners are all handled. If no
valid verdict can be extracted, the result records a parse failure
instead of guessing.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from llm_client import LLMError

PAIRWISE_PROMPT = """You are an impartial judge. Two answers were given to a question. Use ONLY the context below to decide which answer is correct. Do not use outside knowledge. Length and style do not matter; correctness against the context is what counts.

Context:
{context}

Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Reply with JSON only, no other text, exactly in this shape:
{{"winner": "A", "reason": "one short sentence"}}"""

RUBRIC_PROMPT = """You are an impartial judge scoring one answer against the context below. Score each dimension from 1 (poor) to 5 (excellent):
- groundedness: is every claim in the answer supported by the context?
- relevance: does the answer address the question asked?
- completeness: does the answer cover what the question asks for?

Context:
{context}

Question: {question}

Answer:
{answer}

Reply with JSON only, no other text, exactly in this shape:
{{"groundedness": 5, "relevance": 5, "completeness": 5, "reason": "one short sentence"}}"""


def extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from model output.

    Tries, in order: the whole text, a fenced code block, then the
    substring from the first '{' to the last '}'. Returns None if no
    JSON object parses.
    """
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _normalise_winner(value) -> str | None:
    if isinstance(value, str) and value.strip().upper() in ("A", "B"):
        return value.strip().upper()
    return None


@dataclass
class Judgment:
    """One pairwise verdict."""

    winner: Optional[str]  # "A", "B", or None on failure
    reason: str = ""
    raw: str = ""
    parse_ok: bool = False
    error: str = ""  # non-empty if the LLM call itself failed
    latency_s: float = 0.0


@dataclass
class RubricScore:
    """One rubric verdict."""

    scores: dict = field(default_factory=dict)  # groundedness/relevance/completeness
    reason: str = ""
    raw: str = ""
    parse_ok: bool = False
    error: str = ""
    latency_s: float = 0.0


class PairwiseJudge:
    """Asks an LLM which of two answers is better."""

    def __init__(self, client) -> None:
        self.client = client

    def judge(self, question: str, context: str,
              answer_a: str, answer_b: str) -> Judgment:
        prompt = PAIRWISE_PROMPT.format(
            context=context, question=question,
            answer_a=answer_a, answer_b=answer_b)
        started = time.perf_counter()
        try:
            raw = self.client.generate(prompt)
        except LLMError as exc:
            return Judgment(winner=None, error=str(exc),
                            latency_s=time.perf_counter() - started)
        latency = time.perf_counter() - started
        winner, reason = self.parse(raw)
        return Judgment(winner=winner, reason=reason, raw=raw,
                        parse_ok=winner is not None, latency_s=latency)

    @staticmethod
    def parse(raw: str) -> tuple[str | None, str]:
        """Extract (winner, reason) from raw model output."""
        obj = extract_json_object(raw)
        winner, reason = None, ""
        if obj is not None:
            winner = _normalise_winner(obj.get("winner"))
            reason = str(obj.get("reason", ""))[:500]
        if winner is None:
            # Last resort: a bare "winner": "A" fragment.
            match = re.search(r'"winner"\s*:\s*"([ABab])"', raw)
            if match:
                winner = match.group(1).upper()
        return winner, reason


class RubricScorer:
    """Scores one answer 1-5 on groundedness, relevance, completeness."""

    DIMENSIONS = ("groundedness", "relevance", "completeness")

    def __init__(self, client) -> None:
        self.client = client

    def score(self, question: str, context: str, answer: str) -> RubricScore:
        prompt = RUBRIC_PROMPT.format(
            context=context, question=question, answer=answer)
        started = time.perf_counter()
        try:
            raw = self.client.generate(prompt)
        except LLMError as exc:
            return RubricScore(error=str(exc),
                               latency_s=time.perf_counter() - started)
        latency = time.perf_counter() - started
        scores, reason = self.parse(raw)
        return RubricScore(scores=scores, reason=reason, raw=raw,
                           parse_ok=bool(scores), latency_s=latency)

    @classmethod
    def parse(cls, raw: str) -> tuple[dict, str]:
        """Extract ({dimension: int 1-5}, reason); {} if invalid."""
        obj = extract_json_object(raw)
        if obj is None:
            return {}, ""
        scores: dict = {}
        for dim in cls.DIMENSIONS:
            value = obj.get(dim)
            try:
                number = int(value)
            except (TypeError, ValueError):
                return {}, ""
            if not 1 <= number <= 5:
                return {}, ""
            scores[dim] = number
        return scores, str(obj.get("reason", ""))[:500]
