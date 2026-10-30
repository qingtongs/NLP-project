# MultiHop-RAG course project

This project answers all 2,556 questions using the complete 609-document MultiHop-RAG corpus. It uses BM25 passage retrieval with article titles and publishers indexed, clause-based searches, reciprocal-rank fusion, the top 10 passages, and a frozen Qwen3-8B generator. A deterministic evidence gate abstains if a question explicitly names a publisher absent from the corpus. The model has 8.2B total parameters. It never supplies the benchmark's `answer` or `evidence_list` fields to retrieval, generation, or the gate.

The verified complete-run official Accuracy is **77.54% (1,982/2,556)**. Expanding publisher coverage and avoiding a false match on the phrase ��the Australian national team�� improved the previous 77.23% run by 0.31 percentage points. The 90% target has not been reached. The public benchmark was used for diagnostics, so this score is not a hidden-test estimate.

## Sources

- Dataset: [MultiHopRAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG/tree/71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82), revision `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`, ODC-BY.
- Official evaluator: [MultiHop-RAG qa_evaluate.py](https://github.com/yixuantt/MultiHop-RAG/blob/c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8/qa_evaluate.py), commit `c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8`.
- Generator: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), Apache-2.0, base revision `b968826d9c46dd6066d109eabc6255188de91218`. The local [official Q4_K_M GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf) is checked by SHA-256.
- Runtime: [llama.cpp b11804](https://github.com/ggml-org/llama.cpp/releases/tag/b11804), Windows Vulkan build.

## Setup on Windows

Python 3.10+ and an NVIDIA GPU with current Vulkan support are recommended. Use a local Python virtual environment. The model download is about 5.0 GB; `setup_assets.py` downloads and verifies the dataset, model, and Windows runtime. It also downloads the pinned official evaluator. Install `tqdm` for the evaluator.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_assets.py
```

The two course utility scripts `download_eval_files.py` and `prepare_eval.py` remain in `NLP_�ٷ�����̳�/eval_tutorial/` as supplied with the assignment.

## Predict

Start the model server in one terminal:

```powershell
.\runtime\llama-b11804\llama-server.exe -m .\models\Qwen3-8B-Q4_K_M.gguf -ngl 99 -c 8192 -np 1 --flash-attn on --host 127.0.0.1 --port 8080
```

Run the complete benchmark in another terminal. The questions are downloaded by `setup_assets.py`.

```powershell
.\.venv\Scripts\python.exe rag_project.py --corpus corpus.json --questions NLP_�ٷ�����̳�/eval_tutorial/MultiHopRAG.json --model-file models/Qwen3-8B-Q4_K_M.gguf --output-dir run_final --top-k 10 --prompt-mode comparison --no-think
```

The script creates raw `run_final/predictions.json`, `run_final/execution_record.json`, and an append-only JSONL checkpoint. If interrupted, rerun the same command to resume. Each question record stores retrieval queries, ranked passage text and source URLs, actual generator messages and context, output text, token counts, times, failures, and retries.

Apply the evidence sufficiency gate to create the final submission artifacts:

```powershell
.\.venv\Scripts\python.exe evidence_gate.py --corpus corpus.json --questions NLP_�ٷ�����̳�/eval_tutorial/MultiHopRAG.json --raw-record run_final/execution_record.json --output-dir run_final_gated
```

The gate checks names of publishers in the question against the corpus's `source` values. If generation fails with an empty answer after retries, the postprocessor emits an explicit `Insufficient Information` fallback. It saves the raw generator answer, missing publisher names, fallback flag, and final answer under `answer_postprocessing` for every question. It does not inspect reference answers or question types.

## Score

```powershell
cd NLP_�ٷ�����̳�/eval_tutorial
..\..\.venv\Scripts\python.exe prepare_eval.py --predictions ..\..\run_final_gated\predictions.json --queries MultiHopRAG.json --output qa_submission.json
..\..\.venv\Scripts\python.exe qa_evaluate.py --file qa_submission.json --queries MultiHopRAG.json
```

The official evaluator lowercases and whitespace-tokenizes answers. Its printed Precision, Recall, F1, and Accuracy are identical; the course uses overall Accuracy. This public benchmark is suitable for debugging and reported evaluation, and the result is not a hidden-test estimate.

