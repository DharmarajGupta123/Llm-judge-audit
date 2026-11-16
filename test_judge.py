"""Unit tests for the judge audit. No API calls: a fake LLM client.

Run: python3 test_judge.py   (expects: all tests pass, exit 0)
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from unittest import mock

sys.path.insert(0, ".")

from evaluate_judge import Runner, compute_metrics  # noqa: E402
from judge import PairwiseJudge, RubricScorer  # noqa: E402
from llm_client import (  # noqa: E402
    CannedClient,
    LLMError,
    PollinationsClient,
)

FAILURES: list[str] = []


def check(name, condition, detail=""):
    if condition:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name} {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_pairwise_clean_json():
    winner, reason = PairwiseJudge.parse(
        '{"winner": "B", "reason": "B matches the context."}')
    check("pairwise clean json", winner == "B" and "context" in reason)


def test_pairwise_fenced_json():
    winner, _ = PairwiseJudge.parse(
        'Here is my verdict:\n```json\n{"winner": "A", "reason": "x"}\n```')
    check("pairwise fenced json", winner == "A")


def test_pairwise_json_with_prose():
    winner, _ = PairwiseJudge.parse(
        'After comparing both answers, my decision is '
        '{"winner": "B", "reason": "B is correct."} Hope this helps!')
    check("pairwise json with prose", winner == "B")


def test_pairwise_malformed():
    winner, _ = PairwiseJudge.parse(
        "Both answers look reasonable, I cannot decide.")
    check("pairwise malformed -> no winner", winner is None)


def test_pairwise_lowercase_winner():
    winner, _ = PairwiseJudge.parse('{"winner": "b", "reason": "x"}')
    check("pairwise lowercase winner normalised", winner == "B")


def test_pairwise_winner_fragment_fallback():
    # JSON is broken (trailing comma) but the winner fragment survives.
    winner, _ = PairwiseJudge.parse('{"winner": "A", "reason": "x",}')
    check("pairwise fragment fallback", winner == "A")


def test_pairwise_invalid_winner_value():
    winner, _ = PairwiseJudge.parse('{"winner": "both", "reason": "x"}')
    check("pairwise invalid winner rejected", winner is None)


def test_rubric_valid():
    scores, _ = RubricScorer.parse(
        '{"groundedness": 5, "relevance": 4, "completeness": 3, "reason": "x"}')
    check("rubric valid", scores == {"groundedness": 5, "relevance": 4,
                                      "completeness": 3})


def test_rubric_out_of_range():
    scores, _ = RubricScorer.parse(
        '{"groundedness": 6, "relevance": 4, "completeness": 3, "reason": "x"}')
    check("rubric out-of-range rejected", scores == {})


def test_rubric_missing_dimension():
    scores, _ = RubricScorer.parse(
        '{"groundedness": 5, "relevance": 4, "reason": "x"}')
    check("rubric missing dimension rejected", scores == {})


def test_rubric_non_numeric():
    scores, _ = RubricScorer.parse(
        '{"groundedness": "high", "relevance": 4, "completeness": 3}')
    check("rubric non-numeric rejected", scores == {})


# ---------------------------------------------------------------------------
# Judge behaviour with a fake client
# ---------------------------------------------------------------------------

def test_judge_records_latency_and_raw():
    client = CannedClient('{"winner": "A", "reason": "ok"}')
    verdict = PairwiseJudge(client).judge("q?", "ctx", "ans A", "ans B")
    check("judge parses canned response",
          verdict.winner == "A" and verdict.parse_ok
          and verdict.latency_s >= 0 and "winner" in verdict.raw)


def test_judge_llm_error_is_recorded_not_raised():
    class ExplodingClient:
        name = "boom"

        def generate(self, prompt):
            raise LLMError("network down")

    verdict = PairwiseJudge(ExplodingClient()).judge("q?", "ctx", "a", "b")
    check("judge records llm error",
          verdict.winner is None and not verdict.parse_ok
          and "network down" in verdict.error)


def test_canned_client_cycles_and_records_prompts():
    client = CannedClient(["one", "two"])
    outs = [client.generate("p1"), client.generate("p2"), client.generate("p3")]
    check("canned client cycles",
          outs == ["one", "two", "one"] and client.calls == ["p1", "p2", "p3"])


# ---------------------------------------------------------------------------
# Swap logic (Runner.pairwise with a client that always picks A)
# ---------------------------------------------------------------------------

def _runner_with_always_a():
    client = CannedClient('{"winner": "A", "reason": "always A"}')
    handle = io.StringIO()
    return Runner(PairwiseJudge(client), RubricScorer(client),
                  handle, set(), 10)


def test_swap_logic_accuracy():
    runner = _runner_with_always_a()
    normal = runner.pairwise("acc-01-normal", "accuracy", "acc-01", "normal",
                             "q?", "ctx", "correct", "wrong")
    swapped = runner.pairwise("acc-01-swapped", "accuracy", "acc-01", "swapped",
                              "q?", "ctx", "wrong", "correct")
    check("swap logic: always-A is right in normal, wrong in swapped",
          normal["correct_pick"] is True and swapped["correct_pick"] is False)


def test_swap_logic_verbosity():
    runner = _runner_with_always_a()
    normal = runner.pairwise("verb-01-normal", "verbosity", "verb-01", "normal",
                             "q?", "ctx", "short", "long")
    swapped = runner.pairwise("verb-01-swapped", "verbosity", "verb-01", "swapped",
                              "q?", "ctx", "long", "short")
    check("swap logic: always-A picks long only when long is A",
          normal["picked_long"] is False and swapped["picked_long"] is True)


def test_swap_logic_self_pref():
    runner = _runner_with_always_a()
    own_a = runner.pairwise("self-01-ownA", "self_pref", "self-01", "ownA",
                            "q?", "ctx", "own", "para", own_position="A")
    own_b = runner.pairwise("self-01-ownB", "self_pref", "self-01", "ownB",
                            "q?", "ctx", "para", "own", own_position="B")
    check("swap logic: always-A picks own only when own is A",
          own_a["picked_own"] is True and own_b["picked_own"] is False)


# ---------------------------------------------------------------------------
# Metrics math on synthetic records
# ---------------------------------------------------------------------------

def _pairwise_record(item_id, phase, pair_id, order, winner,
                     correct_pick=None, picked_long=None, picked_own=None,
                     own_position=None, parse_ok=True, error=""):
    rec = {"item_id": item_id, "phase": phase, "pair_id": pair_id,
           "order": order, "winner": winner, "parse_ok": parse_ok,
           "error": error, "latency_s": 10.0}
    if correct_pick is not None:
        rec["correct_pick"] = correct_pick
    if picked_long is not None:
        rec["picked_long"] = picked_long
    if picked_own is not None:
        rec["picked_own"] = picked_own
    if own_position is not None:
        rec["own_position"] = own_position
    return rec


def test_metrics_accuracy_and_flips():
    records = [
        # Pair 1: correct in both orders (no flip).
        _pairwise_record("acc-01-normal", "accuracy", "acc-01", "normal",
                         "A", correct_pick=True),
        _pairwise_record("acc-01-swapped", "accuracy", "acc-01", "swapped",
                         "B", correct_pick=True),
        # Pair 2: correct in normal, wrong in swapped (flip).
        _pairwise_record("acc-02-normal", "accuracy", "acc-02", "normal",
                         "A", correct_pick=True),
        _pairwise_record("acc-02-swapped", "accuracy", "acc-02", "swapped",
                         "A", correct_pick=False),
        # Pair 3: wrong in both orders (no flip, consistently wrong).
        _pairwise_record("acc-03-normal", "accuracy", "acc-03", "normal",
                         "B", correct_pick=False),
        _pairwise_record("acc-03-swapped", "accuracy", "acc-03", "swapped",
                         "A", correct_pick=False),
        # Pair 4: parse failure in one order (excluded from flips).
        _pairwise_record("acc-04-normal", "accuracy", "acc-04", "normal",
                         None, parse_ok=False),
        _pairwise_record("acc-04-swapped", "accuracy", "acc-04", "swapped",
                         "B", correct_pick=True),
    ]
    m = compute_metrics(records)
    check("metrics accuracy 4/7",
          m["accuracy"] == {"correct": 4, "total": 7, "rate": 0.5714},
          f"got {m['accuracy']}")
    check("metrics flips 1/3",
          m["position_flips"] == {"flips": 1, "pairs": 3, "rate": 0.3333},
          f"got {m['position_flips']}")
    check("metrics parse failures 1/8",
          m["parse_failures"] == {"failures": 1, "total": 8, "rate": 0.125},
          f"got {m['parse_failures']}")


def test_metrics_verbosity_and_self_pref():
    records = [
        _pairwise_record("verb-01-normal", "verbosity", "verb-01", "normal",
                         "B", picked_long=True),
        _pairwise_record("verb-01-swapped", "verbosity", "verb-01", "swapped",
                         "A", picked_long=True),
        _pairwise_record("verb-02-normal", "verbosity", "verb-02", "normal",
                         "A", picked_long=False),
        _pairwise_record("verb-02-swapped", "verbosity", "verb-02", "swapped",
                         "B", picked_long=False),
        _pairwise_record("self-01-ownA", "self_pref", "self-01", "ownA",
                         "A", picked_own=True, own_position="A"),
        _pairwise_record("self-01-ownB", "self_pref", "self-01", "ownB",
                         "B", picked_own=True, own_position="B"),
        _pairwise_record("self-02-ownA", "self_pref", "self-02", "ownA",
                         "B", picked_own=False, own_position="A"),
        _pairwise_record("self-02-ownB", "self_pref", "self-02", "ownB",
                         "A", picked_own=False, own_position="B"),
    ]
    m = compute_metrics(records)
    check("metrics verbosity 2/4 with 1 both-orders pair",
          m["verbosity"]["rate"] == 0.5
          and m["verbosity"]["both_orders_long"] == 1,
          f"got {m['verbosity']}")
    check("metrics self-preference 2/4",
          m["self_preference"]["rate"] == 0.5
          and m["self_preference"]["own_in_a"] == {"picks": 1, "total": 2},
          f"got {m['self_preference']}")
    check("metrics position A rate 4/8",
          m["position_a_rate"] == {"a_picks": 4, "total": 8, "rate": 0.5},
          f"got {m['position_a_rate']}")


def test_metrics_rubric_means():
    records = [
        {"item_id": "rubric-acc-01-correct", "phase": "rubric",
         "pair_id": "acc-01", "answer_kind": "correct",
         "scores": {"groundedness": 5, "relevance": 5, "completeness": 4},
         "parse_ok": True, "error": "", "latency_s": 9.0},
        {"item_id": "rubric-acc-01-wrong", "phase": "rubric",
         "pair_id": "acc-01", "answer_kind": "wrong",
         "scores": {"groundedness": 1, "relevance": 3, "completeness": 2},
         "parse_ok": True, "error": "", "latency_s": 9.0},
        {"item_id": "rubric-acc-03-wrong", "phase": "rubric",
         "pair_id": "acc-03", "answer_kind": "wrong",
         "scores": {}, "parse_ok": False, "error": "", "latency_s": 9.0},
    ]
    m = compute_metrics(records)
    check("metrics rubric means (parse failure excluded)",
          m["rubric_means"]["correct"] == {"groundedness": 5.0,
                                           "relevance": 5.0,
                                           "completeness": 4.0, "n": 1}
          and m["rubric_means"]["wrong"] == {"groundedness": 1.0,
                                             "relevance": 3.0,
                                             "completeness": 2.0, "n": 1},
          f"got {m['rubric_means']}")


def test_metrics_empty_records():
    m = compute_metrics([])
    check("metrics on empty records do not crash",
          m["accuracy"]["rate"] is None
          and m["position_flips"]["rate"] is None
          and m["avg_latency_s"] is None)


# ---------------------------------------------------------------------------
# Pollinations client (mocked network)
# ---------------------------------------------------------------------------

def test_pollinations_url_and_headers():
    client = PollinationsClient()
    fake_response = io.BytesIO(b'{"winner": "A"}')

    class FakeContext:
        def __enter__(self):
            return fake_response

        def __exit__(self, *args):
            return False

    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["headers"] = dict(request.header_items())
        seen["timeout"] = timeout
        return FakeContext()

    with mock.patch("llm_client.urllib.request.urlopen", fake_urlopen):
        text = client.generate("hello world?")
    check("pollinations url encodes prompt and model",
          seen["url"].startswith("https://text.pollinations.ai/hello%20world%3F")
          and "model=openai" in seen["url"]
          and "private=true" in seen["url"]
          and text == '{"winner": "A"}',
          f"got url={seen.get('url')}")
    check("pollinations sends user-agent",
          "User-agent" in seen["headers"]
          and "llm-judge-audit" in seen["headers"]["User-agent"])


def test_pollinations_retries_then_raises():
    client = PollinationsClient(max_retries=2, backoff_base_s=0.01,
                                min_interval_s=0)
    calls = {"n": 0}

    def failing_urlopen(request, timeout=None):
        calls["n"] += 1
        raise urllib.error.URLError("boom")

    with mock.patch("llm_client.urllib.request.urlopen", failing_urlopen):
        try:
            client.generate("hi")
            raised = False
        except LLMError:
            raised = True
    check("pollinations retries 3 attempts then raises LLMError",
          raised and calls["n"] == 3, f"attempts={calls['n']}")


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for _, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"ERROR {fn.__name__}: {exc}")
            FAILURES.append(fn.__name__)
    print(f"\n{len(tests) - len(FAILURES)} passed, "
          f"{len(FAILURES)} failed, {len(tests)} total")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
