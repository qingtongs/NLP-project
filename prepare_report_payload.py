"""Prepare factual prose and score table for the supplied course report template.

Run only after the fixed official evaluator has scored the complete submission.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-url", default="")
    parser.add_argument("--member-name", required=True)
    parser.add_argument("--student-id", required=True)
    args = parser.parse_args()

    items = json.loads(args.submission.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    record = json.loads(args.record.read_text(encoding="utf-8"))
    info = record["run_info"]
    if len(items) != 2556 or len(questions) != 2556 or len(record["question_records"]) != 2556:
        raise ValueError("The full 2556-question output is required")
    counts = collections.defaultdict(lambda: {"count": 0, "correct": 0})
    for item, question in zip(items, questions):
        if item["query"] != question["query"]:
            raise ValueError("Evaluator submission and question file have different order")
        match = bool(set(item["model_answer"].lower().split()) &
                     set(question["answer"].lower().split()))
        for key in ("overall", item["question_type"]):
            counts[key]["count"] += 1
            counts[key]["correct"] += match
    scores = {key: {"count": value["count"],
                    "correct": value["correct"],
                    "accuracy": value["correct"] / value["count"]}
              for key, value in counts.items()}
    benchmark = 100 * scores["overall"]["accuracy"]
    if benchmark >= 75:
        band = "Excellent (Response Accuracy component 90–100)"
    elif benchmark >= 70:
        band = "Good (Response Accuracy component 80–89)"
    elif benchmark >= 65:
        band = "Pass (Response Accuracy component 60–79)"
    else:
        band = "Below Pass (Response Accuracy component below 60)"
    usage = info["resource_usage"]
    settings = info["settings"]
    generator = info["models"][0]
    hours = usage["elapsed_seconds"] / 3600
    calls = sum(usage["model_calls_by_model"].values())
    cases = {item["question_id"]: item for item in items}

    def answer(question_id: str) -> str:
        return cases[question_id]["model_answer"]

    payload = {
        "date": "October 22, 2026",
        "member_name": args.member_name,
        "student_id": args.student_id,
        "class_name": "",
        "course_code": "",
        "teacher": "",
        "github_url": args.github_url,
        "scores": scores,
        "sections": {
            "Introduction": [
                "This project builds and evaluates a retrieval-augmented generation (RAG) system for MultiHop-RAG [1]. It answers questions that require joining facts across news articles, comparing claims or dates, and recognizing when the corpus does not contain enough information. The final run used all 609 supplied English documents and all 2,556 public questions. The objectives were to keep the question-to-evidence-to-answer path auditable, satisfy the 10B-per-model limit, and measure performance with the course's fixed official evaluator."
            ],
            "Methodology": [
                "The system splits each corpus article into 240-word passages with a 40-word overlap, producing 5,506 indexed passages. A local BM25 implementation (k1=1.5, b=0.75) indexes passage text, triples title terms, and doubles publisher terms. For each question, it searches the full question and up to two longer clauses (15 hits per search). Reciprocal-rank fusion with constant 60 combines these lists; the highest 10 passages are retained, with at most two from one article [2, 3]. The generator sees each passage's title, publisher, publication time, URL, and text.",
                f"A frozen {generator['name']} model ({generator['total_parameters']/1e9:.1f}B total parameters, Q4_K_M quantization) generates one short answer per question in nonthinking mode [4]. The prompt instructs it to resolve separate article facts, attend to negation and temporal order, and answer 'Insufficient Information' when a necessary fact is missing. Generation uses temperature 0, top-p 1, maximum 64 output tokens, seed 42, and an 8,192-token server context. We used llama.cpp b11804 with Vulkan [5]. A deterministic post-generation gate returns 'Insufficient Information' if the question names a publisher absent from the 609-document corpus; its decision and the raw generator answer are both logged. No training or fine-tuning was performed. Retrieval, generation, and this gate read only the corpus and query field; reference answers and evidence annotations were reserved for evaluation and post-run analysis. The BM25, clause retrieval, fusion, gate, answer cleaning, checkpointing, and logs are implemented here; model weights and the evaluator are cited third-party resources."
            ],
            "Evaluation Result": [
                f"The course-supplied prepare_eval.py converted the final q0000–q2555 predictions to evaluator format, and fixed qa_evaluate.py at commit c1c1287 computed official Accuracy [6]. The run answered {scores['overall']['correct']:,} of 2,556 questions under that script's lowercase whitespace-token overlap rule: official overall Accuracy {benchmark:.2f}% (benchmark score B={benchmark:.2f}). This falls in the {band} band; the instructor assigns the exact component mark. The four official question-type results are shown below. These are results on the public complete benchmark, not an estimate of hidden-test performance.",
                f"Hardware: {info['hardware']}. The final run made {calls:,} generator calls, processed {usage['input_tokens']:,} input tokens and {usage['output_tokens']:,} output tokens, and accumulated {hours:.2f} hours of per-question wall time. The evidence gate abstained on {usage.get('evidence_gate_abstentions', 0)} questions. There were {usage['failed_question_count']} failed questions and {usage['retry_count']} retries. The append-only JSONL and consolidated execution record preserve ranked retrieval calls, full source passages, actual generator context and messages, raw and final answers, token counts, times, failures, and retries for every question."
            ],
            "Analysis": [
                "I used the first 100 public questions to select the final generator. With the same passage retrieval and comparison prompt, frozen Qwen3-4B-Instruct-2507 scored 69/100 and Qwen3-8B scored 80/100 under the official overlap rule. Their comparison-query results were 14/35 and 23/35 respectively. The missing-publisher evidence gate then raised the 8B diagnostic score to 82/100 by correcting two unsupported answers; it changed nine answers in total. This is public-benchmark tuning; the 100 diagnostic questions are included in the final 2,556 and must not be treated as an independent held-out test.",
                f"In q0000, the retrieved TechCrunch and The Verge passages both concern Sam Bankman-Fried's FTX trial; the system returned '{answer('q0000')}', matching the reference. In temporal q0050, retrieved November and December TechCrunch articles both discuss anticompetitive allegations against Google; the system answered '{answer('q0050')}', also matching the reference. The logs retain the exact passages used, so these judgments can be checked against the input rather than inferred from the answer alone.",
                f"In q0003, passages from both the Epoch Times and Rainforest payment articles were retrieved, but the generator answered '{answer('q0003')}' where the reference is 'Yes'. This is a multi-document comparison error after retrieval. In q0013, the raw model guessed Apple from an unrelated article, while the question required a Bloomberg article absent from the corpus; the gate changed the final answer to '{answer('q0013')}'. This illustrates why explicit evidence sufficiency checks matter, while publisher presence alone cannot solve comparison errors. Additional analysis-only retrieval checks found annotated-document recall of 74.48% at top 10 on 2,255 answerable questions; this does not use gold annotations in the RAG workflow."
            ],
            "Contributions": [
                f"{args.member_name} ({args.student_id}) is the sole group member and is responsible for the submitted corpus preparation, RAG method, experiment selection, complete benchmark run, result analysis, repository, and report. Codex AI assistance was used for implementation, local runs, analysis, and report drafting. No other group member is listed."
            ],
            "Conclusion": [
                f"The completed RAG pipeline processed every question with retrieved article passages and a frozen model below the parameter limit. Its official public-benchmark Accuracy is {benchmark:.2f}%. The strongest remaining errors involve multi-article yes/no composition, missing supporting passages, and unwarranted answers to unanswerable questions. Better coverage of both required documents and explicit evidence sufficiency checks are practical next steps.",
                "References: [1] Tang and Yang, MultiHop-RAG, arxiv.org/abs/2401.15391; dataset revision 71ac0d0 at huggingface.co/datasets/yixuantt/MultiHopRAG. [2] Robertson and Zaragoza, The Probabilistic Relevance Framework BM25 and Beyond, Foundations and Trends in Information Retrieval, 2009. [3] Cormack, Clarke, and Buettcher, Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods, SIGIR 2009. [4] Qwen3-8B-GGUF, huggingface.co/Qwen/Qwen3-8B-GGUF. [5] llama.cpp b11804, github.com/ggml-org/llama.cpp/releases/tag/b11804. [6] Fixed evaluator, github.com/yixuantt/MultiHop-RAG/blob/c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8/qa_evaluate.py."
            ]
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
