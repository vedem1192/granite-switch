#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run answerability accuracy experiments: prompting vs adapter.

This script tests answerability accuracy using either:
1. Prompting the base model (baseline)
2. Using the answerability adapter (Granite Switch)

Usage:
    # Prompting baseline
    python run_answerability_accuracy.py \
      --mode prompting \
      --model ibm-granite/granite-4.1-30b \
      --output-dir ./accuracy_results/answerability/prompting

    # Adapter (aLoRA)
    python run_answerability_accuracy.py \
      --mode adapter \
      --model ibm-granite/granite-switch-4.1-30b \
      --output-dir ./accuracy_results/answerability/adapter

    # With multiple seeds
    python run_answerability_accuracy.py \
      --mode adapter \
      --seeds 3 \
      --output-dir ./accuracy_results/answerability/adapter
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["prompting", "adapter"],
        required=True,
        help="Test mode: prompting (baseline) or adapter",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to use (HF repo or local path)",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=Path("rag_101_queries.txt"),
        help="File with test queries (default: rag_101_queries.txt)",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path("/data//granite-switch-benchmark/granite-switch/tutorials/scripts/comparison/alora_vs_lora_race/alora_vs_lora_race/govt_chroma"),
        help="Path to ChromaDB corpus",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of seeds to run (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default: 8000)",
    )
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Launch vLLM server automatically",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=10240,
        help="vLLM max-model-len (default: 10240)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)",
    )
    return parser.parse_args()


def load_queries(queries_file: Path) -> tuple[list[str], list[str]]:
    """Load queries and their expected labels (answerable/unanswerable)."""
    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    answerable_queries = []
    unanswerable_queries = []

    in_unanswerable_section = False

    with open(queries_file) as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                # Check if this is the unanswerable section marker
                if "unanswerable" in line.lower() or "off-topic" in line.lower():
                    in_unanswerable_section = True
                continue

            # Add to appropriate list based on section
            if in_unanswerable_section:
                unanswerable_queries.append(line)
            else:
                answerable_queries.append(line)

    print(f"Loaded {len(answerable_queries)} answerable queries")
    print(f"Loaded {len(unanswerable_queries)} unanswerable queries")

    return answerable_queries, unanswerable_queries


def start_vllm_server(model: str, port: int, max_model_len: int, log_file: Path):
    """Start a vLLM server."""
    print(f"[SERVER] Starting vLLM on port {port}...")
    print(f"         Model: {model}")
    print(f"         Log: {log_file}")

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
    ]

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"         PID: {proc.pid}")

    # Wait for server to be ready
    import urllib.request

    health_url = f"http://localhost:{port}/health"
    start = time.time()
    timeout = 300

    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    elapsed = int(time.time() - start)
                    print(f"[READY] Server ready in {elapsed}s")
                    return proc
        except Exception:
            pass
        time.sleep(5)

    raise RuntimeError(f"Server failed to start after {timeout}s")


def run_answerability_prompting(
    query: str, documents: list[str], base_url: str, model: str
) -> str:
    """Test answerability using prompting (baseline)."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="unused")

    # Create prompt for answerability check
    docs_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])

    prompt = f"""Given the following documents and a question, determine if the question can be answered using ONLY the information in the documents.

Documents:
{docs_text}

Question: {query}

Answer with ONLY one word: "answerable" or "unanswerable"

Answer:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        answer = response.choices[0].message.content.strip().lower()

        # Parse response
        if "answerable" in answer and "unanswerable" not in answer:
            return "answerable"
        elif "unanswerable" in answer:
            return "unanswerable"
        else:
            # Default to unanswerable if unclear
            return "unanswerable"

    except Exception as e:
        print(f"ERROR: {e}")
        return "unanswerable"


def run_answerability_adapter(
    query: str, documents: list[str], base_url: str, model: str
) -> str:
    """Test answerability using the adapter."""
    from mellea.backends.openai import OpenAIBackend
    from mellea.stdlib.components import Document as MelleaDocument
    from mellea.stdlib.components.intrinsic import rag
    from mellea.stdlib.context import ChatContext

    backend = OpenAIBackend(model_id=model, base_url=base_url, api_key="unused")
    backend.register_embedded_adapter_model(model)

    docs = [MelleaDocument(doc_id=str(i), text=text) for i, text in enumerate(documents)]
    ctx = ChatContext()

    try:
        result = rag.check_answerability(query, docs, ctx, backend)
        return result.lower()
    except Exception as e:
        print(f"ERROR: {e}")
        return "unanswerable"


def retrieve_documents(query: str, chroma_path: Path, top_k: int) -> list[str]:
    """Retrieve top-k documents for a query."""
    from granite_switch.tutorials.chroma_loader import load_or_build_govt_chroma

    # Load collection with correct embedding function (same as rag_101.ipynb)
    # This is done per-query which is inefficient, but keeps the function simple
    collection = load_or_build_govt_chroma(
        chroma_path=str(chroma_path),
        load_only_tutorial_docs=False,
        query_device="cpu",
    )

    results = collection.query(query_texts=[query], n_results=top_k)
    documents = results.get("documents", [[]])[0]
    return documents


def run_experiment_seed(
    answerable_queries: list[str],
    unanswerable_queries: list[str],
    mode: str,
    model: str,
    base_url: str,
    chroma_path: Path,
    top_k: int,
    seed: int,
) -> dict:
    """Run one seed of the experiment."""
    print(f"\n[SEED {seed}] Running {len(answerable_queries) + len(unanswerable_queries)} queries...")

    results = []
    correct = 0
    total = 0

    # Test answerable queries
    for i, query in enumerate(answerable_queries, 1):
        documents = retrieve_documents(query, chroma_path, top_k)

        if mode == "prompting":
            prediction = run_answerability_prompting(query, documents, base_url, model)
        else:
            prediction = run_answerability_adapter(query, documents, base_url, model)

        is_correct = prediction == "answerable"
        correct += is_correct
        total += 1

        results.append({
            "query": query,
            "expected": "answerable",
            "predicted": prediction,
            "correct": is_correct,
            "num_documents": len(documents),
        })

        if (i % 10) == 0:
            print(f"  Progress: {i}/{len(answerable_queries)} answerable queries")

    # Test unanswerable queries
    for i, query in enumerate(unanswerable_queries, 1):
        documents = retrieve_documents(query, chroma_path, top_k)

        if mode == "prompting":
            prediction = run_answerability_prompting(query, documents, base_url, model)
        else:
            prediction = run_answerability_adapter(query, documents, base_url, model)

        is_correct = prediction == "unanswerable"
        correct += is_correct
        total += 1

        results.append({
            "query": query,
            "expected": "unanswerable",
            "predicted": prediction,
            "correct": is_correct,
            "num_documents": len(documents),
        })

        if (i % 10) == 0:
            print(f"  Progress: {i}/{len(unanswerable_queries)} unanswerable queries")

    accuracy = correct / total if total > 0 else 0

    # Compute per-class metrics
    tp = sum(1 for r in results if r["expected"] == "answerable" and r["predicted"] == "answerable")
    fp = sum(1 for r in results if r["expected"] == "unanswerable" and r["predicted"] == "answerable")
    tn = sum(1 for r in results if r["expected"] == "unanswerable" and r["predicted"] == "unanswerable")
    fn = sum(1 for r in results if r["expected"] == "answerable" and r["predicted"] == "unanswerable")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"[SEED {seed}] Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"          Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    return {
        "seed": seed,
        "mode": mode,
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
        "total": total,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "results": results,
    }


def aggregate_results(seed_results: list[dict]) -> dict:
    """Aggregate statistics across seeds."""
    if not seed_results:
        return {}

    accuracies = [r["accuracy"] for r in seed_results]
    precisions = [r["precision"] for r in seed_results]
    recalls = [r["recall"] for r in seed_results]
    f1s = [r["f1"] for r in seed_results]

    aggregate = {
        "num_seeds": len(seed_results),
        "mode": seed_results[0]["mode"],
        "model": seed_results[0]["model"],
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "accuracy_min": float(np.min(accuracies)),
        "accuracy_max": float(np.max(accuracies)),
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "seeds_data": seed_results,
    }

    return aggregate


def main():
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "model": args.model,
        "seeds": args.seeds,
        "queries_file": str(args.queries_file),
        "chroma_path": str(args.chroma_path),
        "top_k": args.top_k,
    }
    config_file = args.output_dir / "experiment_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 80)
    print("ANSWERABILITY ACCURACY EXPERIMENT")
    print("=" * 80)
    print(f"Mode:        {args.mode}")
    print(f"Model:       {args.model}")
    print(f"Seeds:       {args.seeds}")
    print(f"Output dir:  {args.output_dir}")
    print("=" * 80)

    # Load queries
    answerable_queries, unanswerable_queries = load_queries(args.queries_file)

    # Start server if needed
    server_proc = None
    if args.launch_server:
        log_file = args.output_dir / "vllm_server.log"
        server_proc = start_vllm_server(args.model, args.port, args.max_model_len, log_file)

    base_url = f"http://localhost:{args.port}/v1"

    # Run experiments for each seed
    seed_results = []
    for seed in range(1, args.seeds + 1):
        result = run_experiment_seed(
            answerable_queries,
            unanswerable_queries,
            args.mode,
            args.model,
            base_url,
            args.chroma_path,
            args.top_k,
            seed,
        )
        seed_results.append(result)

        # Save seed result
        seed_file = args.output_dir / f"results_seed{seed}.json"
        with open(seed_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[SAVED] {seed_file}")

    # Stop server if we started it
    if server_proc:
        print("\n[SERVER] Stopping vLLM...")
        server_proc.terminate()
        server_proc.wait()

    # Compute aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    aggregate = aggregate_results(seed_results)

    if aggregate:
        aggregate_file = args.output_dir / "aggregate_statistics.json"
        with open(aggregate_file, "w") as f:
            json.dump(aggregate, f, indent=2)
        print(f"[SAVED] {aggregate_file}")

        print(f"\n📊 Results (mean ± std across {aggregate['num_seeds']} seeds)")
        print("─" * 80)
        print(f"Accuracy:  {aggregate['accuracy_mean']:.3f} ± {aggregate['accuracy_std']:.3f}")
        print(f"Precision: {aggregate['precision_mean']:.3f} ± {aggregate['precision_std']:.3f}")
        print(f"Recall:    {aggregate['recall_mean']:.3f} ± {aggregate['recall_std']:.3f}")
        print(f"F1:        {aggregate['f1_mean']:.3f} ± {aggregate['f1_std']:.3f}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
