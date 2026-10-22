"""Analyze completed predictions against the public benchmark after inference."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def match(prediction: str, reference: str) -> bool:
    # The fixed official qa_evaluate.py uses whitespace-token intersection.
    return bool(set(prediction.lower().split()) & set(reference.lower().split()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--show-errors", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        records = records[: args.limit]
    reference = json.loads(args.questions.read_text(encoding="utf-8"))
    summary = collections.defaultdict(lambda: collections.Counter())
    examples = []
    for row in records:
        index = int(row["question_id"][1:])
        question = reference[index]
        good = match(row["model_answer"], question["answer"])
        for group in ("overall", question["question_type"]):
            summary[group]["total"] += 1
            summary[group]["correct"] += good
        if not good and len(examples) < args.show_errors:
            examples.append((row["question_id"], question["query"], question["answer"], row["model_answer"]))
    for group, values in summary.items():
        print(group, values["correct"], "/", values["total"],
              f"= {values['correct'] / values['total']:.4f}")
    for question_id, question, reference_answer, prediction in examples:
        print(f"\n{question_id}: {question}\n  reference: {reference_answer}\n  predicted: {prediction}")


if __name__ == "__main__":
    main()
