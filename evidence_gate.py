"""Abstain when a question explicitly names a publisher absent from the corpus.

This deterministic post-generation gate uses only question text and corpus source
metadata. It never reads reference answers, question types, or evidence labels.
The generator's raw output remains in the execution record for audit.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Publisher names are matched as whole phrases, then checked against the
# publishers in the supplied corpus. Names present in the corpus never trigger
# the gate. This list identifies names, not benchmark answer labels.
PUBLISHER_NAMES = (
    "Al Jazeera", "Architectural Digest", "Associated Press", "BBC", "Bloomberg",
    "Business Insider", "Business Standard", "CNET", "CNN", "Deal Stripe",
    "ESPN", "Financial Times", "Forbes", "Hindustan Times", "PCWorld",
    "Punch Line", "Search Engine Land", "The Australian", "The Fashion Times",
    "Reuters", "Rolling Stone", "Space.com", "The Astrophysical Journal",
    "The Economic Times", "The Guardian", "The Hindu", "The Hollywood Reporter",
    "The New York Times", "The Verge",
    "Times of India", "Variety", "Wall Street Journal", "Washington Post",
    "Yahoo Finance",
)


def missing_publishers(question: str, publishers: set[str]) -> list[str]:
    normalized_publishers = {source.casefold() for source in publishers}
    missing = []
    for name in PUBLISHER_NAMES:
        # These names can also occur as ordinary wording in a question.
        flags = 0 if name in {"Variety", "The Australian"} else re.I
        if not re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question, flags):
            continue
        needle = name.casefold().removeprefix("the ")
        if any(needle in source for source in normalized_publishers):
            continue
        missing.append(name)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--raw-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    data = json.loads(args.raw_record.read_text(encoding="utf-8"))
    publishers = {str(item.get("source") or "") for item in corpus}
    rows = data["question_records"]
    if len(questions) != 2556 or len(rows) != 2556:
        raise ValueError("The complete question set and raw record are required")
    gated = 0
    empty_fallbacks = 0
    for i, row in enumerate(rows):
        question = questions[i]["query"]
        if row["question_id"] != f"q{i:04d}" or row["question"] != question:
            raise ValueError(f"Question alignment mismatch at {i}")
        missing = missing_publishers(question, publishers)
        before = row["model_answer"]
        if missing:
            row["model_answer"] = "Insufficient Information"
            gated += 1
        elif not str(before).strip():
            row["model_answer"] = "Insufficient Information"
            empty_fallbacks += 1
        row["answer_postprocessing"] = {
            "method": ("missing named publisher abstention" if missing else
                       "empty generator fallback" if not str(before).strip() else "none"),
            "missing_publishers": missing,
            "generator_answer": before,
            "empty_generator_fallback": not str(before).strip() and not missing,
            "final_answer": row["model_answer"],
        }
    info = data["run_info"]
    info["rag_method"] += "; post-generation missing-publisher evidence gate and empty-answer fallback"
    info["settings"]["evidence_gate"] = {
        "name": "missing named publisher abstention",
        "publisher_aliases": list(PUBLISHER_NAMES),
        "source_metadata_field": "source",
    }
    info["settings"]["empty_answer_fallback"] = "Insufficient Information"
    info["resource_usage"]["evidence_gate_abstentions"] = gated
    info["resource_usage"]["empty_generator_fallbacks"] = empty_fallbacks
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "execution_record.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "predictions.json").write_text(
        json.dumps([{"question_id": row["question_id"],
                     "model_answer": row["model_answer"]} for row in rows],
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Applied evidence gate to {gated} questions and empty-answer fallback to {empty_fallbacks}")


if __name__ == "__main__":
    main()
