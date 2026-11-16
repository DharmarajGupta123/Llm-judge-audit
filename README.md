# LLM-as-a-Judge Audit

A pairwise LLM judge that scores answers — audited for three biases
(position bias, verbosity bias, self-preference) against
**known-correct preferences**, so no human labelling was needed to know
which answer should win.

## What it is

An LLM judge is only useful if its verdicts track answer quality rather
than surface features. This project builds a judge and then measures
exactly that, on test pairs constructed so the right winner is known by
design:

- **Accuracy set (10 pairs):** a grounded answer vs a hallucinated or
  unsupported one, about small fictional documents (a leave policy, a
  headphones manual, a store's shipping page). The grounded answer must
  win.
- **Position-bias check:** every accuracy pair is judged twice, with
  A/B swapped. A verdict that changes with the order is a position
  flip.
- **Verbosity set (6 pairs):** a short correct answer vs a long,
  confident, rambling WRONG answer. The short one must win. Judged
  swapped too, so position bias can be separated from verbosity bias.
- **Self-preference set (6 items):** the judge model's own generated
  answer vs a hand-written paraphrase of the SAME content. Neither
  should win systematically; a steady preference for its own wording
  is the bias signal.
- **Rubric sanity check:** the single-answer rubric scorer (1–5 on
  groundedness, relevance, completeness) scores 4 correct and 4 wrong
  answers, to confirm the rubric separates right from wrong.

## How the judge works

`judge.py` has two modes, both driven by a pluggable LLM client
(`llm_client.py`):

- **Pairwise:** the judge gets the question, the context, and two
  answers, and returns JSON `{"winner": "A"|"B", "reason": "..."}`.
  The prompt instructs it to use only the context, and states that
  length and style do not matter — so if it still picks the long wrong
  answer, that is a real bias signal, not a prompt oversight.
- **Rubric:** one answer scored 1–5 on groundedness, relevance, and
  completeness, returned as JSON.

Parsing is defensive (fenced JSON, surrounding prose, lowercase
winners all handled); anything that still does not parse is recorded
as a parse failure, never guessed.

## Providers

- **Pollinations (default, free, no key):** the judge runs on
  `text.pollinations.ai` with `model=openai`. All results below were
  measured with this provider.
- **Gemini:** pass `--provider gemini` with a `GOOGLE_API_KEY` set, to
  run the same audit against Gemini (`gemini-2.0-flash`, temperature
  0). Needs `langchain-google-genai` installed.

## Measured results

Judge model: Pollinations `model=openai`, measured 2026-11-17.
44 pairwise judgments + 8 rubric scores.

| Metric | Result |
|---|---|
| Accuracy (correct-winner rate) | 20/20 = 100% |
| Position flip rate | 0/10 = 0% |
| Position A pick rate (50% = neutral) | 27/44 = 61.4% |
| Verbosity bias (picked long wrong) | 0/12 = 0% |
| Self-preference (picked own wording, 50% = neutral) | 7/12 = 58.3% (see findings — confounded by position) |
| JSON parse failures | 0/44 = 0% |
| Call errors (API failures) | 0 |
| Avg latency per call | 13.1 s |
| Rubric mean, correct answers (n=4) | groundedness 5.0, relevance 5.0, completeness 5.0 |
| Rubric mean, wrong answers (n=4) | groundedness 1.0, relevance 4.0, completeness 3.25 |

**What the numbers say:**

- **The judge is reliable when answers differ in quality.** 20/20 on
  accuracy with zero position flips, and 0/12 on verbosity bias: it
  picked the short correct answer over the long wrong one in both
  orders, every time.
- **Position bias shows up only on ties.** The 61.4% position-A rate
  comes entirely from the self-preference phase, where both answers
  had equal content: the judge picked A in 11 of 12 of those
  judgments. In the accuracy and verbosity phases (where quality
  differed), it picked A in exactly 16 of 32 judgments — no position
  effect at all.
- **The self-preference result is inconclusive, not positive.** The
  headline 7/12 (58.3%) is fully explained by that position-A
  default: the model's own wording won 6/6 when it sat in position A
  and only 1/6 in position B. The judge's own reasons confirm it —
  one reads "Both answers match the context; either could be correct
  but A was chosen arbitrarily." A cleaner self-preference test would
  need a judge without a position tie-break, or many more items.
- **The rubric's groundedness dimension does the work.** Correct
  answers scored 5.0 on all three dimensions; wrong answers scored
  1.0 on groundedness but 4.0 on relevance and 3.25 on completeness —
  the wrong answers were on-topic and reasonably complete, just
  unsupported by the context. For catching hallucinations,
  groundedness is the dimension that matters.

Full per-item results (including the judge's reasons and raw replies)
are in `results/results.json`; the incremental log is
`results/results.jsonl`.

## Run it

```bash
# Full audit (~10-20 min, 52 free API calls)
env -u NO_PROXY -u no_proxy python3 evaluate_judge.py

# Same audit against Gemini (needs GOOGLE_API_KEY)
env -u NO_PROXY -u no_proxy python3 evaluate_judge.py --provider gemini

# Harness check with canned responses (no network)
python3 evaluate_judge.py --dry-run --out-dir /tmp/dryrun-results

# Unit tests (fake LLM client, no API calls)
python3 test_judge.py
```

Results are saved incrementally: every finished item is appended to
`results/results.jsonl`, so an interrupted run loses nothing —
re-running resumes where it stopped (failed API calls are retried;
parse failures are kept as results).

## Files

- `llm_client.py` — provider abstraction (Pollinations + Gemini),
  retries/timeouts, pacing.
- `judge.py` — pairwise judge + rubric scorer, defensive JSON parsing.
- `test_sets.py` — the fictional documents and all constructed pairs.
- `evaluate_judge.py` — runs the audits, prints the table, writes
  `results/results.json` + `results/results.jsonl`.
- `test_judge.py` — unit tests with a fake LLM client.
- `generate_self_answers.py` — one-off setup script that produced the
  judge model's own answers for the self-preference set.
- `requirements.txt` — standard library only (Gemini provider optional).

## Method notes and limitations

- **Why no human labels:** every pair's correct winner is known by
  construction (the answer is either in the quoted context or it is
  not), so the audit measures the judge against ground truth without
  human annotation. The documented next step is comparing the judge's
  verdicts against human preferences on open-ended answers, where
  "correct" is a matter of judgement rather than lookup.
- **Small test set:** 10 accuracy pairs, 6 verbosity pairs, 6
  self-preference items. The rates above are indicative, not precise
  estimates — with n=6, one item swinging changes a rate by ~8 points
  (judgments, not items, are the denominator where noted).
- **Self-preference is the weakest leg — and the run proved it:**
  the paraphrases were written by hand to preserve content, but the
  judge's position-A tie-break dominated the result (see findings),
  so this run cannot separate self-preference from position bias.
  Treat the 58.3% as inconclusive, not as evidence of
  self-preference.
- **Judge non-determinism:** Pollinations `model=openai` is not
  temperature-controlled through this API, so re-running the audit
  may give slightly different numbers. The test sets are fixed, so
  the audit is re-runnable and comparable.
- **Fictional documents:** the contexts are short, clean, and
  unambiguous by design. Real RAG contexts are messier, and judge
  accuracy there is likely lower.
