"""Generate the judge model's OWN answers for the self-preference set.

One-off setup script (not part of the audit run): it asks the judge
model to answer the SELF_PREF_QUESTIONS from their contexts, and prints
the answers as JSON. Those answers are then hand-paraphrased (same
content, different style) and both versions are stored in
test_sets.SELF_PREF_ITEMS, so the audit itself is fully reproducible
without extra generation calls.

Usage:
    env -u NO_PROXY -u no_proxy python3 generate_self_answers.py
"""

from __future__ import annotations

import json
import sys

from llm_client import get_client
from test_sets import DOCS, SELF_PREF_QUESTIONS

GENERATION_PROMPT = """Answer the question using ONLY the context below. Answer in one or two sentences.

Context:
{context}

Question: {question}

Answer:"""


def main() -> int:
    client = get_client("pollinations")
    print(f"Generating own-answers with {client.name} ...", file=sys.stderr)
    results = []
    for item in SELF_PREF_QUESTIONS:
        prompt = GENERATION_PROMPT.format(
            context=DOCS[item["doc"]], question=item["question"])
        answer = client.generate(prompt)
        results.append({**item, "own": answer})
        print(f"  {item['id']}: {answer!r}", file=sys.stderr)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
