"""Run the judge audits and report the measured numbers.

Phases (judge calls in brackets, Pollinations default):
  1. Accuracy     -- 10 pairs x 2 orders (A/B swapped)         [20]
  2. Verbosity    -- 6 pairs x 2 orders                         [12]
  3. Self-preference -- 6 items x 2 orders                      [12]
  4. Rubric sanity -- 4 pairs, correct + wrong answer scored    [8]

Total: 52 calls, roughly 10-20 minutes at ~10-20s per call.

Every completed item is appended to results/results.jsonl as soon as it
finishes, so an interrupted run loses nothing; re-running resumes from
the JSONL (failed API calls are retried, parse failures are kept as
results -- re-rolling them would bias the parse-failure rate down).
A summary + all records are written to results/results.json at the end.

Usage:
    env -u NO_PROXY -u no_proxy python3 evaluate_judge.py
    env -u NO_PROXY -u no_proxy python3 evaluate_judge.py --provider gemini
    python3 evaluate_judge.py --dry-run        # no network, harness check
    python3 evaluate_judge.py --limit 2        # 2 pairs per phase (quick test)
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from judge import PairwiseJudge, RubricScorer
from llm_client import CannedClient, get_client
from test_sets import (
    ACCURACY_PAIRS,
    DOCS,
    RUBRIC_PAIR_IDS,
    SELF_PREF_ITEMS,
    VERBOSITY_PAIRS,
)

HERE = Path(__file__).resolve().parent
DRY_RUN_PAIRWISE = '{"winner": "A", "reason": "dry run"}'
DRY_RUN_RUBRIC = ('{"groundedness": 5, "relevance": 5, "completeness": 5, '
                  '"reason": "dry run"}')
RAW_TRUNCATE = 2000


# ---------------------------------------------------------------------------
# Incremental results store
# ---------------------------------------------------------------------------

def load_records(jsonl_path: Path) -> list[dict]:
    """Load prior results; drop API-error records (they get retried)."""
    records: list[dict] = []
    if not jsonl_path.exists():
        return records
    kept, dropped, corrupt = 0, 0, 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                corrupt += 1
                continue
            if record.get("error"):
                dropped += 1
                continue
            records.append(record)
            kept += 1
    if dropped or corrupt:
        print(f"Resuming: kept {kept} records, dropping {dropped} errored "
              f"and {corrupt} corrupt lines (they will be re-run).")
    elif kept:
        print(f"Resuming: {kept} records already done.")
    return records


def rewrite_jsonl(jsonl_path: Path, records: list[dict]) -> None:
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_record(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


# ---------------------------------------------------------------------------
# Metrics (pure function over records, so tests can check the math)
# ---------------------------------------------------------------------------

def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def compute_metrics(records: list[dict]) -> dict:
    pairwise = [r for r in records
                if r["phase"] in ("accuracy", "verbosity", "self_pref")]
    judged = [r for r in pairwise if r.get("parse_ok")]

    # Accuracy: correct-winner rate over successful accuracy judgments.
    acc = [r for r in judged if r["phase"] == "accuracy"]
    acc_correct = sum(1 for r in acc if r.get("correct_pick"))

    # Position flips: pairs whose content-winner changed between orders.
    flips = flippable = 0
    by_pair: dict[str, dict[str, dict]] = {}
    for r in acc:
        by_pair.setdefault(r["pair_id"], {})[r["order"]] = r
    for orders in by_pair.values():
        if "normal" in orders and "swapped" in orders:
            flippable += 1
            if orders["normal"]["correct_pick"] != orders["swapped"]["correct_pick"]:
                flips += 1

    # Position A pick rate across all successful pairwise judgments.
    a_picks = sum(1 for r in judged if r.get("winner") == "A")

    # Verbosity: how often the long (wrong) answer won.
    verb = [r for r in judged if r["phase"] == "verbosity"]
    long_picks = sum(1 for r in verb if r.get("picked_long"))
    verb_normal = [r for r in verb if r["order"] == "normal"]  # long in B
    verb_swapped = [r for r in verb if r["order"] == "swapped"]  # long in A
    long_normal = sum(1 for r in verb_normal if r.get("picked_long"))
    long_swapped = sum(1 for r in verb_swapped if r.get("picked_long"))
    verb_both_long = 0
    verb_pairs: dict[str, dict[str, dict]] = {}
    for r in verb:
        verb_pairs.setdefault(r["pair_id"], {})[r["order"]] = r
    for orders in verb_pairs.values():
        if ("normal" in orders and "swapped" in orders
                and orders["normal"].get("picked_long")
                and orders["swapped"].get("picked_long")):
            verb_both_long += 1

    # Self-preference: how often the model's own wording won.
    selfp = [r for r in judged if r["phase"] == "self_pref"]
    own_picks = sum(1 for r in selfp if r.get("picked_own"))
    own_in_a = [r for r in selfp if r["own_position"] == "A"]
    own_in_b = [r for r in selfp if r["own_position"] == "B"]
    own_a_picks = sum(1 for r in own_in_a if r.get("picked_own"))
    own_b_picks = sum(1 for r in own_in_b if r.get("picked_own"))

    # Parse failures and call errors (over all pairwise calls attempted).
    parse_fails = sum(1 for r in pairwise
                      if not r.get("parse_ok") and not r.get("error"))
    call_errors = sum(1 for r in pairwise if r.get("error"))

    latencies = [r["latency_s"] for r in records
                 if isinstance(r.get("latency_s"), (int, float))]
    avg_latency = (round(sum(latencies) / len(latencies), 1)
                   if latencies else None)

    # Rubric: mean scores for correct vs wrong answers.
    rubric = [r for r in records
              if r["phase"] == "rubric" and r.get("parse_ok")]
    rubric_means: dict[str, dict[str, float | None]] = {}
    for kind in ("correct", "wrong"):
        subset = [r for r in rubric if r["answer_kind"] == kind]
        rubric_means[kind] = {}
        for dim in ("groundedness", "relevance", "completeness"):
            values = [r["scores"][dim] for r in subset if dim in r["scores"]]
            rubric_means[kind][dim] = (
                round(sum(values) / len(values), 2) if values else None)
        rubric_means[kind]["n"] = len(subset)

    return {
        "accuracy": {"correct": acc_correct, "total": len(acc),
                     "rate": _rate(acc_correct, len(acc))},
        "position_flips": {"flips": flips, "pairs": flippable,
                           "rate": _rate(flips, flippable)},
        "position_a_rate": {"a_picks": a_picks, "total": len(judged),
                            "rate": _rate(a_picks, len(judged))},
        "verbosity": {"long_picks": long_picks, "total": len(verb),
                      "rate": _rate(long_picks, len(verb)),
                      "long_in_a": {"picks": long_swapped,
                                    "total": len(verb_swapped)},
                      "long_in_b": {"picks": long_normal,
                                    "total": len(verb_normal)},
                      "both_orders_long": verb_both_long,
                      "pairs": len(verb_pairs)},
        "self_preference": {"own_picks": own_picks, "total": len(selfp),
                            "rate": _rate(own_picks, len(selfp)),
                            "own_in_a": {"picks": own_a_picks,
                                         "total": len(own_in_a)},
                            "own_in_b": {"picks": own_b_picks,
                                         "total": len(own_in_b)}},
        "parse_failures": {"failures": parse_fails, "total": len(pairwise),
                           "rate": _rate(parse_fails, len(pairwise))},
        "call_errors": call_errors,
        "avg_latency_s": avg_latency,
        "rubric_means": rubric_means,
        "n_pairwise_judgments": len(pairwise),
        "n_records": len(records),
    }


def _pct(rate: float | None) -> str:
    return f"{rate:.1%}" if rate is not None else "n/a"


def print_report(metrics: dict, provider_name: str) -> None:
    a = metrics["accuracy"]
    f = metrics["position_flips"]
    p = metrics["position_a_rate"]
    v = metrics["verbosity"]
    s = metrics["self_preference"]
    pf = metrics["parse_failures"]
    print()
    print(f"Judge audit results (provider: {provider_name})")
    print("=" * 60)
    print(f"Accuracy (correct-winner rate):     {a['correct']:>2}/{a['total']:<2} = {_pct(a['rate'])}")
    print(f"Position flip rate:                 {f['flips']:>2}/{f['pairs']:<2} = {_pct(f['rate'])}  (pairs where the verdict changed when A/B swapped)")
    print(f"Position A pick rate:               {p['a_picks']:>2}/{p['total']:<2} = {_pct(p['rate'])}  (50% = no position preference)")
    print(f"Verbosity bias (picked long wrong): {v['long_picks']:>2}/{v['total']:<2} = {_pct(v['rate'])}")
    print(f"  - long answer in position A:      {v['long_in_a']['picks']:>2}/{v['long_in_a']['total']:<2}")
    print(f"  - long answer in position B:      {v['long_in_b']['picks']:>2}/{v['long_in_b']['total']:<2}")
    print(f"  - long picked in BOTH orders:     {v['both_orders_long']:>2}/{v['pairs']:<2} pairs")
    print(f"Self-preference (picked own words): {s['own_picks']:>2}/{s['total']:<2} = {_pct(s['rate'])}  (50% = no self-preference)")
    print(f"  - own wording in position A:      {s['own_in_a']['picks']:>2}/{s['own_in_a']['total']:<2}")
    print(f"  - own wording in position B:      {s['own_in_b']['picks']:>2}/{s['own_in_b']['total']:<2}")
    print(f"JSON parse failures:                {pf['failures']:>2}/{pf['total']:<2} = {_pct(pf['rate'])}")
    print(f"Call errors (API failures):         {metrics['call_errors']}")
    avg_lat = metrics["avg_latency_s"]
    print(f"Avg latency per call:               "
          f"{f'{avg_lat} s' if avg_lat is not None else 'n/a'}")
    r = metrics["rubric_means"]
    for kind in ("correct", "wrong"):
        m = r[kind]
        print(f"Rubric mean ({kind}, n={m['n']}): "
              f"groundedness {m['groundedness']}, relevance {m['relevance']}, "
              f"completeness {m['completeness']}")


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

class Runner:
    """Runs phases, appending each finished item to the JSONL."""

    def __init__(self, judge, scorer, handle, done_ids, total) -> None:
        self.judge = judge
        self.scorer = scorer
        self.handle = handle
        self.done_ids = set(done_ids)
        self.total = total
        self.count = len(done_ids)

    def _record(self, record: dict, note: str) -> None:
        self.count += 1
        record["latency_s"] = round(record.get("latency_s", 0.0), 1)
        if record.get("raw"):
            record["raw"] = record["raw"][:RAW_TRUNCATE]
        append_record(self.handle, record)
        print(f"[{self.count}/{self.total}] {record['item_id']}: {note} "
              f"({record['latency_s']}s)", flush=True)

    def pairwise(self, item_id, phase, pair_id, order, question, context,
                 answer_a, answer_b, **extra) -> dict | None:
        if item_id in self.done_ids:
            return None
        verdict = self.judge.judge(question, context, answer_a, answer_b)
        record = {"item_id": item_id, "phase": phase, "pair_id": pair_id,
                  "order": order, "winner": verdict.winner,
                  "parse_ok": verdict.parse_ok, "error": verdict.error,
                  "latency_s": verdict.latency_s, "reason": verdict.reason,
                  "raw": verdict.raw, **extra}
        if verdict.parse_ok:
            if phase == "accuracy":
                record["correct_pick"] = (
                    (verdict.winner == "A" and order == "normal")
                    or (verdict.winner == "B" and order == "swapped"))
                note = f"winner={verdict.winner} correct={record['correct_pick']}"
            elif phase == "verbosity":
                record["picked_long"] = (
                    (verdict.winner == "B" and order == "normal")
                    or (verdict.winner == "A" and order == "swapped"))
                note = f"winner={verdict.winner} picked_long={record['picked_long']}"
            else:  # self_pref
                own_pos = extra["own_position"]
                record["picked_own"] = verdict.winner == own_pos
                note = f"winner={verdict.winner} picked_own={record['picked_own']}"
        elif verdict.error:
            note = f"CALL ERROR: {verdict.error[:100]}"
        else:
            note = "PARSE FAILURE"
        self._record(record, note)
        return record

    def rubric(self, item_id, pair_id, answer_kind, question, context,
               answer) -> dict | None:
        if item_id in self.done_ids:
            return None
        result = self.scorer.score(question, context, answer)
        record = {"item_id": item_id, "phase": "rubric", "pair_id": pair_id,
                  "answer_kind": answer_kind, "scores": result.scores,
                  "parse_ok": result.parse_ok, "error": result.error,
                  "latency_s": result.latency_s, "reason": result.reason,
                  "raw": result.raw}
        note = (f"scores={result.scores}" if result.parse_ok
                else ("CALL ERROR: " + result.error[:100] if result.error
                      else "PARSE FAILURE"))
        self._record(record, note)
        return record


def run_all(judge, scorer, handle, done_ids, limit=None, skip_rubric=False):
    """Run every phase. Returns the list of NEW records."""
    new_records: list[dict] = []
    acc_pairs = ACCURACY_PAIRS[:limit] if limit else ACCURACY_PAIRS
    verb_pairs = VERBOSITY_PAIRS[:limit] if limit else VERBOSITY_PAIRS
    self_items = SELF_PREF_ITEMS[:limit] if limit else SELF_PREF_ITEMS
    rubric_ids = RUBRIC_PAIR_IDS[:limit] if limit else RUBRIC_PAIR_IDS
    total = (len(acc_pairs) * 2 + len(verb_pairs) * 2
             + len(self_items) * 2 + (0 if skip_rubric else len(rubric_ids) * 2))
    runner = Runner(judge, scorer, handle, done_ids, total)

    for pair in acc_pairs:
        ctx = DOCS[pair["doc"]]
        for order, (a, b) in (("normal", (pair["correct"], pair["wrong"])),
                              ("swapped", (pair["wrong"], pair["correct"]))):
            rec = runner.pairwise(f"{pair['id']}-{order}", "accuracy",
                                  pair["id"], order, pair["question"], ctx, a, b)
            if rec:
                new_records.append(rec)

    for pair in verb_pairs:
        ctx = DOCS[pair["doc"]]
        for order, (a, b) in (("normal", (pair["short_correct"], pair["long_wrong"])),
                              ("swapped", (pair["long_wrong"], pair["short_correct"]))):
            rec = runner.pairwise(f"{pair['id']}-{order}", "verbosity",
                                  pair["id"], order, pair["question"], ctx, a, b)
            if rec:
                new_records.append(rec)

    for item in self_items:
        ctx = DOCS[item["doc"]]
        for order, (a, b, own_pos) in (
                ("ownA", (item["own"], item["paraphrase"], "A")),
                ("ownB", (item["paraphrase"], item["own"], "B"))):
            rec = runner.pairwise(f"{item['id']}-{order}", "self_pref",
                                  item["id"], order, item["question"], ctx,
                                  a, b, own_position=own_pos)
            if rec:
                new_records.append(rec)

    if not skip_rubric:
        by_id = {p["id"]: p for p in ACCURACY_PAIRS}
        for pair_id in rubric_ids:
            pair = by_id[pair_id]
            ctx = DOCS[pair["doc"]]
            for kind, answer in (("correct", pair["correct"]),
                                 ("wrong", pair["wrong"])):
                rec = runner.rubric(f"rubric-{pair_id}-{kind}", pair_id, kind,
                                    pair["question"], ctx, answer)
                if rec:
                    new_records.append(rec)
    return new_records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the LLM-judge bias audits.")
    parser.add_argument("--provider", default="pollinations",
                        choices=["pollinations", "gemini"])
    parser.add_argument("--model", default=None,
                        help="model name (default: pollinations=openai, gemini=gemini-2.0-flash)")
    parser.add_argument("--out-dir", default=str(HERE / "results"))
    parser.add_argument("--dry-run", action="store_true",
                        help="canned responses, no network calls")
    parser.add_argument("--skip-rubric", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="max pairs per phase (quick test)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    json_path = out_dir / "results.json"

    if args.dry_run:
        judge = PairwiseJudge(CannedClient(DRY_RUN_PAIRWISE))
        scorer = RubricScorer(CannedClient(DRY_RUN_RUBRIC))
        provider_name = "dry-run (canned)"
    else:
        kwargs = {"model": args.model} if args.model else {}
        client = get_client(args.provider, **kwargs)
        judge = PairwiseJudge(client)
        scorer = RubricScorer(client)
        provider_name = client.name

    prior = load_records(jsonl_path)
    done_ids = {r["item_id"] for r in prior}
    rewrite_jsonl(jsonl_path, prior)  # drops errored/corrupt lines

    print(f"Judge audit starting (provider: {provider_name}).")
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        new_records = run_all(judge, scorer, handle, done_ids,
                              limit=args.limit, skip_rubric=args.skip_rubric)

    all_records = prior + new_records
    metrics = compute_metrics(all_records)
    summary = {
        "provider": provider_name,
        "run_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
        "records": all_records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print_report(metrics, provider_name)
    print(f"\nFull results: {json_path}")
    print(f"Incremental log: {jsonl_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
