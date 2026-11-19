#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""RAG 101: retrieve passages for a query and check answerability (script version of rag_101.ipynb).

Usage:
    python tutorials/scripts/reference/rag_101.py --base-url http://localhost:8000/v1
    python tutorials/scripts/reference/rag_101.py --launch-server
    python tutorials/scripts/reference/rag_101.py -q "How do I renew a passport?"
    python tutorials/scripts/reference/rag_101.py --queries-file queries.txt --parallel 8
    python tutorials/scripts/reference/rag_101.py -i
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import re
import shutil
import signal
import statistics
import sys
import time
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("rag_101")

DEFAULT_MODEL = "ibm-granite/granite-switch-4.1-3b-preview"
DEFAULT_PORT = 8000

DEFAULT_QUERIES = [
    "How do I check the status of my federal tax refund?",
    "What's the best recipe for sourdough bread?",
]

_VLLM_LOG_PREFIX = re.compile(
    r"^\(\w+ pid=\d+\)\s*(?:(?:DEBUG|INFO|WARNING|ERROR)\s+[\d\-]+ [\d:]+ \[[^\]]+\]\s*)?"
)

_CAUSE_PATTERN = re.compile(
    r"\b\w*(?:Error|Exception)\b:|not found|No such file|out of memory|not enough|"
    r"insufficient|not supported|doesn't support|Falling back",
    re.IGNORECASE,
)

_FIX_HINTS = [
    (
        re.compile(r"nvcc.*not found", re.IGNORECASE),
        "No CUDA compiler (nvcc). Set VLLM_USE_FLASHINFER_SAMPLER=0 so vLLM "
        "samples with PyTorch instead of compiling FlashInfer kernels.",
    ),
    (
        re.compile(r"Python\.h", re.IGNORECASE),
        "Python headers missing. Rebuild the env on uv's Python: rm -rf .venv && "
        "uv sync --extra tutorials --python-preference only-managed",
    ),
    (
        re.compile(r"out of memory|not enough KV cache", re.IGNORECASE),
        "GPU memory too small for this context. Lower --max-model-len.",
    ),
    (
        re.compile(r"bfloat16", re.IGNORECASE),
        "GPU lacks bfloat16 (e.g. T4). Try --dtype float16, or use an L4/A100.",
    ),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the rag_101 tutorial (corpus + answerability) as a script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    server = p.add_argument_group("vLLM server")
    server.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL"),
        help="OpenAI-compatible endpoint. Defaults to $VLLM_BASE_URL, "
        "else http://localhost:<port>/v1.",
    )
    server.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL_NAME", DEFAULT_MODEL),
        help="Model name as served by vLLM (GET /v1/models).",
    )
    server.add_argument(
        "--adapter-source",
        default=os.environ.get("GRANITE_SWITCH_SOURCE"),
        help="Hub repo id or local path with the adapters' I/O configs "
        "(defaults to --model).",
    )
    server.add_argument("--port", type=int, default=DEFAULT_PORT)
    server.add_argument(
        "--launch-server",
        action="store_true",
        help="Start vLLM for --model before running (needs a GPU); "
        "stopped again on exit.",
    )
    server.add_argument("--max-model-len", type=int, default=10240)
    server.add_argument(
        "--dtype",
        default=None,
        help="Passed to vLLM as --dtype (e.g. float16 on a T4, which lacks bfloat16).",
    )
    server.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help="With --launch-server: requests vLLM may batch together. "
        "Keep >= --parallel, or parallel requests just queue.",
    )
    server.add_argument("--server-timeout", type=int, default=900, help="Seconds.")
    server.add_argument("--server-log", default="vllm_server.log")

    corpus = p.add_argument_group("corpus")
    corpus.add_argument("--chroma-path", default="./govt_chroma")
    corpus.add_argument("--jsonl-path", default="./govt.jsonl")
    corpus.add_argument(
        "--full-corpus",
        action="store_true",
        help="Embed all ~49k passages instead of the ~140-passage tutorial subset "
        "(slow; delete --chroma-path first if it already holds the subset).",
    )
    corpus.add_argument(
        "--top-k", type=int, default=5, help="Passages retrieved per query."
    )

    run = p.add_argument_group("run")
    run.add_argument(
        "-q",
        "--query",
        action="append",
        dest="queries",
        help="Question to check (repeatable). Defaults to the notebook's two queries.",
    )
    run.add_argument(
        "--queries-file",
        type=Path,
        help="Text file with one question per line (blank lines and # comments skipped).",
    )
    run.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Answerability requests in flight at once; vLLM batches them on the GPU.",
    )
    run.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="After the batch, keep everything loaded and prompt for more questions.",
    )
    run.add_argument(
        "--show-passages",
        type=int,
        default=3,
        help="How many retrieved passages to print per query.",
    )
    run.add_argument("--output", type=Path, help="Write all results as JSON here.")
    run.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Also show mellea / HTTP / library INFO logs.",
    )
    return p.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    """Configure timestamped stdout logging; silence libraries unless verbose."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    if verbose:
        return
    os.environ.setdefault("MELLEA_LOG_LEVEL", "WARNING")
    for name in ("httpx", "chromadb", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


def section(title: str) -> None:
    log.info("")
    log.info("── %s %s", title, "─" * max(0, 66 - len(title)))


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path.cwd() / ".env", override=False)


def load_corpus(args: argparse.Namespace):
    """Build or load the ChromaDB collection of government passages."""
    from granite_switch.tutorials.chroma_loader import load_or_build_govt_chroma

    started = time.monotonic()
    collection = load_or_build_govt_chroma(
        chroma_path=args.chroma_path,
        jsonl_path=args.jsonl_path,
        load_only_tutorial_docs=not args.full_corpus,
    )
    log.info(
        "Corpus ready: %s passages in %s (%.1fs)",
        f"{collection.count():,}",
        args.chroma_path,
        time.monotonic() - started,
    )
    return collection


def find_failure_causes(log_file: str, max_lines: int = 12) -> list[str]:
    """Return the deduplicated lines of a vLLM log that explain a startup failure."""
    try:
        lines = Path(log_file).read_text(errors="replace").splitlines()
    except OSError as exc:
        return [f"could not read {log_file}: {exc}"]

    causes: list[str] = []
    for raw in lines:
        line = _VLLM_LOG_PREFIX.sub("", raw).strip()
        is_stack_frame = line.startswith(("File ", "raise ", "^", "~"))
        if is_stack_frame or not _CAUSE_PATTERN.search(line) or line in causes:
            continue
        causes.append(line)
    return causes[:max_lines]


def report_server_failure(log_file: str) -> None:
    causes = find_failure_causes(log_file)
    log.error("vLLM failed to start. Cause lines from %s:", log_file)
    for line in causes:
        log.error("  | %s", _clip(line, 300))

    text = "\n".join(causes)
    for pattern, hint in _FIX_HINTS:
        if pattern.search(text):
            log.error("Likely fix: %s", hint)
            break


def _server_is_up(port: int) -> bool:
    try:
        url = f"http://localhost:{port}/v1/models"
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except OSError:
        return False


def wait_until_ready(proc, port: int, timeout: int, log_file: str) -> bool:
    """Poll until the server answers; return False if it exits or times out."""
    from granite_switch.tutorials.vllm_server import _current_stage

    started = time.monotonic()
    last_stage, last_logged = None, 0.0
    while (elapsed := time.monotonic() - started) < timeout:
        if proc.poll() is not None:
            log.error("vLLM exited (code %s) after %.0fs", proc.returncode, elapsed)
            return False
        if _server_is_up(port):
            return True

        stage = _current_stage(log_file)
        if stage != last_stage or elapsed - last_logged >= 30:
            log.info("  [%4.0fs] %s...", elapsed, stage)
            last_stage, last_logged = stage, elapsed
        time.sleep(3)

    log.error("vLLM still not ready after %ds", timeout)
    return False


def _use_torch_sampler_if_no_nvcc() -> None:
    """Disable the FlashInfer sampler when nvcc is unavailable."""
    if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ:
        return
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if shutil.which("nvcc") or Path(cuda_home, "bin", "nvcc").exists():
        return
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    log.info("nvcc not found: setting VLLM_USE_FLASHINFER_SAMPLER=0 (PyTorch sampler)")


def start_server(args: argparse.Namespace) -> None:
    """Launch vLLM in the background and block until it is ready."""
    from granite_switch.tutorials.vllm_server import launch_vllm

    _use_torch_sampler_if_no_nvcc()

    extra_args = ["--dtype", args.dtype] if args.dtype else []
    proc = launch_vllm(
        model=args.model,
        port=args.port,
        log_file=args.server_log,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        extra_args=extra_args,
    )

    def stop() -> None:
        if proc.poll() is None:
            log.info("Stopping vLLM server (pid %d)", proc.pid)
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()

    atexit.register(stop)

    started = time.monotonic()
    if not wait_until_ready(proc, args.port, args.server_timeout, args.server_log):
        report_server_failure(args.server_log)
        sys.exit(f"vLLM server did not start; full log: {args.server_log}")
    log.info("vLLM server ready in %.0fs", time.monotonic() - started)


def connect_backend(base_url: str, model: str, adapter_source: str):
    """Create a mellea OpenAIBackend for vLLM with the embedded adapters registered."""
    from mellea.backends.openai import OpenAIBackend

    backend = OpenAIBackend(model_id=model, base_url=base_url, api_key="unused")
    backend.register_embedded_adapter_model(adapter_source)
    adapters = backend.list_adapters()
    log.info("Registered %d adapters: %s", len(adapters), ", ".join(adapters))
    return backend


@dataclass
class Passage:
    doc_id: str
    title: str
    url: str
    distance: float
    text: str


@dataclass
class QueryResult:
    query: str
    verdict: str
    retrieve_seconds: float
    model_seconds: float
    passages: list[Passage] = field(default_factory=list)


def retrieve(collection, query: str, top_k: int) -> list[Passage]:
    """Return the top-K passages from ChromaDB, most similar first."""
    hits = collection.query(query_texts=[query], n_results=top_k)
    return [
        Passage(
            doc_id=doc_id,
            title=(meta or {}).get("title", ""),
            url=(meta or {}).get("url", ""),
            distance=float(distance),
            text=" ".join(text.split()),
        )
        for doc_id, meta, distance, text in zip(
            hits["ids"][0],
            hits["metadatas"][0],
            hits["distances"][0],
            hits["documents"][0],
        )
    ]


def check_answerability(query: str, passages: list[Passage], backend) -> str:
    """Return 'answerable' or 'unanswerable' from the answerability adapter."""
    from mellea.stdlib.components import Document
    from mellea.stdlib.components.intrinsic import rag
    from mellea.stdlib.context import ChatContext

    documents = [Document(doc_id=str(i), text=p.text) for i, p in enumerate(passages)]
    return rag.check_answerability(query, documents, ChatContext(), backend)


def _timed(fn, *args):
    started = time.monotonic()
    value = fn(*args)
    return value, time.monotonic() - started


def run_batch(
    queries: list[str], collection, backend, top_k: int, parallel: int
) -> Iterator[QueryResult]:
    """Retrieve for each query, then check answerability with `parallel` workers."""
    retrieved = [_timed(retrieve, collection, q, top_k) for q in queries]

    def answer(i: int) -> QueryResult:
        passages, retrieve_s = retrieved[i]
        verdict, model_s = _timed(check_answerability, queries[i], passages, backend)
        return QueryResult(queries[i], verdict, retrieve_s, model_s, passages)

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        yield from pool.map(answer, range(len(queries)))


def report(result: QueryResult, show_passages: int) -> None:
    mark = "✅" if result.verdict == "answerable" else "🚫"
    log.info(
        "%s %-12s %s  (model %.2fs)",
        mark,
        result.verdict.upper(),
        result.query,
        result.model_seconds,
    )
    for rank, p in enumerate(result.passages[:show_passages], start=1):
        log.info(
            "     #%d  dist=%.3f  %s", rank, p.distance, _clip(p.title or p.doc_id, 90)
        )
        log.info("         %s", _clip(p.text, 140))


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def summarize(results: list[QueryResult], wall_seconds: float, parallel: int) -> None:
    answerable = sum(r.verdict == "answerable" for r in results)
    model = sorted(r.model_seconds for r in results)
    log.info("%d / %d queries answerable from the corpus", answerable, len(results))
    log.info(
        "%d queries in %.1fs wall time = %.2f queries/s  (parallel=%d)",
        len(results),
        wall_seconds,
        len(results) / wall_seconds,
        parallel,
    )
    log.info(
        "Model latency per query: median %.2fs, max %.2fs  |  retrieval avg %.3fs",
        statistics.median(model),
        model[-1],
        statistics.fmean(r.retrieve_seconds for r in results),
    )


def read_queries(args: argparse.Namespace) -> list[str]:
    """Collect queries from -q and --queries-file, falling back to the defaults."""
    queries = list(args.queries or [])
    if args.queries_file:
        for line in args.queries_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    if not queries and not args.interactive:
        queries = DEFAULT_QUERIES
    return queries


def interactive_loop(
    collection, backend, args: argparse.Namespace
) -> list[QueryResult]:
    """Prompt for questions until an empty line, 'quit', or Ctrl-D."""
    log.info("Interactive mode: type a question, empty line or Ctrl-D to quit.")
    results = []
    while True:
        try:
            query = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in ("", "q", "quit", "exit"):
            break
        for result in run_batch([query], collection, backend, args.top_k, 1):
            report(result, args.show_passages)
            results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = parse_args(argv)
    setup_logging(args.verbose)

    base_url = args.base_url or f"http://localhost:{args.port}/v1"
    adapter_source = args.adapter_source or args.model
    queries = read_queries(args)

    log.info("vLLM endpoint : %s", base_url)
    log.info("Model         : %s", args.model)
    log.info(
        "Corpus        : %s (%s)",
        args.chroma_path,
        "full" if args.full_corpus else "tutorial subset",
    )

    section("1 · Corpus")
    collection = load_corpus(args)

    if args.launch_server:
        section("2 · Launch vLLM server")
        start_server(args)

    section("3 · Connect mellea")
    backend = connect_backend(base_url, args.model, adapter_source)

    results: list[QueryResult] = []
    if queries:
        section(f"4 · Retrieve + answerability ({len(queries)} queries)")
        started = time.monotonic()
        for result in run_batch(
            queries, collection, backend, args.top_k, args.parallel
        ):
            report(result, args.show_passages)
            results.append(result)
        wall_seconds = time.monotonic() - started

        section("Summary")
        summarize(results, wall_seconds, args.parallel)

    if args.interactive:
        section("Interactive")
        results += interactive_loop(collection, backend, args)

    if args.output:
        args.output.write_text(json.dumps([asdict(r) for r in results], indent=2))
        log.info("Results written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
