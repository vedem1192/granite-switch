#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run multi-seed latency experiments for aLoRA vs LoRA comparison.

This script orchestrates multiple runs of the benchmark with different seeds,
saves results per seed, generates aggregate statistics, and creates visualizations.

Usage:
    # Run both aLoRA and LoRA with 3 seeds each
    python run_latency_experiments.py --seeds 3 --runs 50 --concurrency 8

    # Run only aLoRA
    python run_latency_experiments.py --seeds 3 --server-type alora

    # Custom model paths
    python run_latency_experiments.py --seeds 3 \
        --alora-model /path/to/alora \
        --lora-model /path/to/lora

    # Dry run (show what would be executed)
    python run_latency_experiments.py --seeds 3 --dry-run
"""

import argparse
import json
import os
import shutil
import signal
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
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds to run (default: 3)",
    )
    parser.add_argument(
        "--runs",
        "-n",
        type=int,
        default=50,
        help="Number of conversations per seed (default: 50)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=8,
        help="Max concurrent requests per server (default: 8)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Number of documents to retrieve per query (default: 10)",
    )
    parser.add_argument(
        "--server-type",
        choices=["alora", "lora", "both"],
        default="both",
        help="Which server type to benchmark (default: both)",
    )
    parser.add_argument(
        "--alora-model",
        default="/data/vdemers/granite-switch-benchmark/granite-switch-alora-prefer",
        help="Path to aLoRA model checkpoint",
    )
    parser.add_argument(
        "--lora-model",
        default="/data/vdemers/granite-switch-benchmark/granite-switch-lora-only",
        help="Path to LoRA model checkpoint",
    )
    parser.add_argument(
        "--alora-port",
        type=int,
        default=8111,
        help="Port for aLoRA vLLM server (default: 8111)",
    )
    parser.add_argument(
        "--lora-port",
        type=int,
        default=8112,
        help="Port for LoRA vLLM server (default: 8112)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./latency_results"),
        help="Directory to save results (default: ./latency_results)",
    )
    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=Path("/data/vdemers/granite-switch-benchmark/granite-switch/tutorials/scripts/comparison/alora_vs_lora_race/bench_pipeline_race.py"),
        help="Path to bench_pipeline_race.py",
    )
    parser.add_argument(
        "--race-dir",
        type=Path,
        default=Path("/data/vdemers/granite-switch-benchmark/granite-switch/tutorials/scripts/comparison/alora_vs_lora_race"),
        help="Directory containing race scripts and data",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="vLLM --max-model-len (default: 32768)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--skip-server-management",
        action="store_true",
        help="Assume servers are already running (for debugging)",
    )
    parser.add_argument(
        "--keep-servers-running",
        action="store_true",
        help="Don't terminate servers after each seed (faster but uses more memory)",
    )
    return parser.parse_args()


class VLLMServer:
    """Manages a vLLM server process."""

    def __init__(self, model_path: str, port: int, log_file: Path, max_model_len: int):
        self.model_path = model_path
        self.port = port
        self.log_file = log_file
        self.max_model_len = max_model_len
        self.proc = None

    def start(self, dry_run: bool = False):
        """Start the vLLM server."""
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_path,
            "--port",
            str(self.port),
            "--max-model-len",
            str(self.max_model_len),
        ]

        print(f"[START] Launching vLLM server on port {self.port}")
        print(f"        Model: {self.model_path}")
        print(f"        Log: {self.log_file}")

        if dry_run:
            print(f"        Command: {' '.join(cmd)}")
            return

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "w") as f:
            self.proc = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, text=True
            )

        print(f"        PID: {self.proc.pid}")

        # Wait for server to be ready
        if not self._wait_for_ready(timeout=300):
            self._print_tail_log(lines=30)
            raise RuntimeError(f"vLLM server on port {self.port} failed to start")

        print(f"[READY] Server on port {self.port} is ready")

    def _wait_for_ready(self, timeout: int = 300) -> bool:
        """Poll /health endpoint until ready."""
        import urllib.request

        health_url = f"http://localhost:{self.port}/health"
        start = time.time()

        while time.time() - start < timeout:
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        elapsed = int(time.time() - start)
                        print(f"        Ready in {elapsed}s")
                        return True
            except Exception:
                pass

            elapsed = int(time.time() - start)
            if elapsed % 30 == 0:  # Print every 30s
                print(f"        Waiting... {elapsed}s elapsed")
            time.sleep(5)

        return False

    def _print_tail_log(self, lines: int = 20):
        """Print last N lines of the log file."""
        try:
            with open(self.log_file) as f:
                tail = f.readlines()[-lines:]
                print(f"\n[ERROR] Last {lines} lines of {self.log_file}:")
                for line in tail:
                    print(f"  | {line.rstrip()}")
        except Exception as e:
            print(f"[ERROR] Could not read log file: {e}")

    def stop(self, dry_run: bool = False):
        """Stop the vLLM server."""
        if dry_run or self.proc is None:
            return

        if self.proc.poll() is None:
            print(f"[STOP] Terminating server on port {self.port} (pid {self.proc.pid})")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print(f"[KILL] Force killing server on port {self.port}")
                self.proc.kill()
                self.proc.wait()

        # Wait a bit for GPU memory to be freed
        time.sleep(5)


def kill_stale_vllm_processes():
    """Kill any existing vLLM processes."""
    result = subprocess.run(
        ["pgrep", "-f", "vllm.entrypoints"], capture_output=True, text=True
    )
    pids = [p for p in result.stdout.strip().split("\n") if p]

    if pids:
        print(f"[CLEANUP] Killing stale vLLM processes: {pids}")
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(5)
    else:
        print("[CLEANUP] No stale vLLM processes found")


def _extract_server_only(mixed_data: dict, server_name: str) -> dict:
    """Extract only one server's data from the mixed results."""
    if "servers" not in mixed_data or server_name not in mixed_data["servers"]:
        raise RuntimeError(f"Server {server_name} not found in results")

    server_data = mixed_data["servers"][server_name]

    # Create a clean structure with just this server
    clean = {
        "metadata": mixed_data.get("metadata", {}),
        "server_name": server_name,
        "model": server_data.get("model", ""),
        "adapter_tech": server_data.get("adapter_tech", {}),
        "wall_total": server_data.get("wall_total", 0),
        "conv_walls": server_data.get("conv_walls", []),
        "conversations": server_data.get("conversations", []),
        "exits": server_data.get("exits", {}),
        "step_medians": server_data.get("step_medians", {}),
    }

    return clean


def _filter_events_for_server(mixed_events: dict, server_name: str) -> dict:
    """Filter events to include only one server."""
    filtered_events = []

    for event in mixed_events.get("events", []):
        if event.get("srv") == server_name:
            filtered_events.append(event)

    clean = {
        "metadata": mixed_events.get("metadata", {}),
        "server_name": server_name,
        "events": filtered_events,
    }

    return clean


def run_benchmark(
    benchmark_script: Path,
    race_dir: Path,
    server_name: str,
    server_port: int,
    model_path: str,
    seed: int,
    runs: int,
    concurrency: int,
    top_k: int,
    output_dir: Path,
    dry_run: bool = False,
) -> Path:
    """Run the benchmark for one seed and return the results file path."""
    # Determine server flag
    server_flag = "--alora-model" if "alora" in server_name.lower() else "--lora-model"

    cmd = [
        sys.executable,
        str(benchmark_script),
        "--mode",
        "sequential",
        "--server",
        server_name,
        server_flag,
        model_path,
        "--no-live",  # Disable Rich display
        "-n",
        str(runs),
        "-c",
        str(concurrency),
        "-k",
        str(top_k),
    ]

    print(f"\n[BENCHMARK] Running seed {seed} for {server_name}")
    print(f"            Runs: {runs}, Concurrency: {concurrency}, Top-K: {top_k}")

    if dry_run:
        print(f"            Command: {' '.join(cmd)}")
        return output_dir / f"race_results_seed{seed}.json"

    # Change to race directory (benchmark script expects to run from there)
    original_cwd = Path.cwd()
    os.chdir(race_dir)

    try:
        # Clean old results before running (prevents mixed data)
        race_results_path = race_dir / "race_results.json"
        race_events_path = race_dir / "race_events.json"
        race_results_path.unlink(missing_ok=True)
        race_events_path.unlink(missing_ok=True)
        print(f"            Cleaned old race results")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            print(f"[ERROR] Benchmark failed with code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"Benchmark failed for seed {seed}")

        # Load and extract only this server's data
        if not race_results_path.exists():
            raise RuntimeError(f"Expected results file not found: {race_results_path}")

        with open(race_results_path) as f:
            mixed_data = json.load(f)

        with open(race_events_path) as f:
            mixed_events = json.load(f)

        # Extract only this server's data
        clean_results = _extract_server_only(mixed_data, server_name)
        clean_events = _filter_events_for_server(mixed_events, server_name)

        # Save cleaned data
        output_results = output_dir / f"race_results_seed{seed}.json"
        output_events = output_dir / f"race_events_seed{seed}.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_results, "w") as f:
            json.dump(clean_results, f, indent=2)
        print(f"[SAVED] Results (clean): {output_results}")

        with open(output_events, "w") as f:
            json.dump(clean_events, f, indent=2)
        print(f"[SAVED] Events (clean): {output_events}")

        return output_results.resolve()

    finally:
        os.chdir(original_cwd)


def aggregate_results(result_files: list[Path], output_path: Path):
    """Aggregate statistics across multiple seeds."""
    print(f"\n[AGGREGATE] Computing statistics across {len(result_files)} seeds")

    all_results = []
    for fpath in result_files:
        with open(fpath) as f:
            data = json.load(f)
            all_results.append(data)

    if not all_results:
        print("[WARNING] No results to aggregate")
        return

    # Extract key metrics per seed (now using clean structure)
    metrics_per_seed = []
    for seed_idx, data in enumerate(all_results):
        metrics = {
            "seed": seed_idx + 1,
            "total_conversations": len(data.get("conversations", [])),
            "total_wall_time_s": data.get("wall_total", 0),
        }

        # Extract per-step latencies
        conversations = data.get("conversations", [])
        if conversations:
            step_latencies = {}
            for conv in conversations:
                for turn in conv.get("turns", []):
                    for step in turn.get("steps", []):
                        step_name = step.get("step", "unknown")
                        duration = step.get("duration_s", 0)
                        if step_name not in step_latencies:
                            step_latencies[step_name] = []
                        step_latencies[step_name].append(duration)

            # Compute median per step
            for step_name, latencies in step_latencies.items():
                metrics[f"median_{step_name}_s"] = float(np.median(latencies))

        metrics_per_seed.append(metrics)

    # Compute aggregate statistics
    aggregate = {
        "timestamp": datetime.now().isoformat(),
        "num_seeds": len(result_files),
        "seeds_data": metrics_per_seed,
    }

    # Compute mean and std across seeds for each metric
    if metrics_per_seed:
        first_metrics = metrics_per_seed[0]
        for key in first_metrics:
            if key == "seed":
                continue
            values = [m.get(key, 0) for m in metrics_per_seed]
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_std"] = float(np.std(values))
            aggregate[f"{key}_min"] = float(np.min(values))
            aggregate[f"{key}_max"] = float(np.max(values))

    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"[SAVED] Aggregate statistics: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    if "total_wall_time_s_mean" in aggregate:
        mean_time = aggregate["total_wall_time_s_mean"]
        std_time = aggregate["total_wall_time_s_std"]
        print(f"Total wall time: {mean_time:.1f} ± {std_time:.1f}s")

    print("\nPer-step median latencies (mean ± std across seeds):")
    for key in sorted(aggregate.keys()):
        if key.startswith("median_") and key.endswith("_s_mean"):
            step_name = key.replace("median_", "").replace("_s_mean", "")
            mean_lat = aggregate[key]
            std_lat = aggregate.get(f"median_{step_name}_s_std", 0)
            print(f"  {step_name:20s}: {mean_lat:.3f} ± {std_lat:.3f}s")
    print("=" * 70)


def generate_html_reports(
    result_files: list[Path], race_dir: Path, output_dir: Path, server_label: str
):
    """Generate HTML visualization reports for each seed."""
    print(f"\n[HTML] Skipping HTML generation - templates expect 2-server race format")
    print(f"       Use the JSON files directly or run analyze_latency_results.py")
    print(f"       for paper-ready statistics.")

    # Note: The HTML templates from sample_run/ are designed for simultaneous
    # 2-server races. Since we now save clean single-server data, we'd need
    # different templates. For now, skip HTML generation and use JSON directly.


def _embed_json_in_html(
    template_path: Path,
    json_path: Path,
    output_path: Path,
    var_name: str,
    marker: str,
):
    """Embed JSON data into an HTML template."""
    import re

    with open(json_path) as f:
        data = json.load(f)

    with open(template_path) as f:
        html = f.read()

    html = re.sub(
        rf"const {var_name} = .*?; // {re.escape(marker)}",
        f"const {var_name} = {json.dumps(data)}; // {marker}",
        html,
        flags=re.DOTALL,
    )

    with open(output_path, "w") as f:
        f.write(html)


def main():
    args = parse_args()

    # Validate paths
    if not args.benchmark_script.exists():
        sys.exit(f"ERROR: Benchmark script not found: {args.benchmark_script}")
    if not args.race_dir.exists():
        sys.exit(f"ERROR: Race directory not found: {args.race_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SETUP] Output directory: {args.output_dir}")

    # Save experiment configuration
    config = {
        "timestamp": datetime.now().isoformat(),
        "seeds": args.seeds,
        "runs": args.runs,
        "concurrency": args.concurrency,
        "top_k": args.top_k,
        "server_type": args.server_type,
        "alora_model": args.alora_model,
        "lora_model": args.lora_model,
        "max_model_len": args.max_model_len,
    }
    config_path = args.output_dir / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[SAVED] Experiment config: {config_path}")

    # Determine which servers to run
    servers = []
    if args.server_type in ["alora", "both"]:
        servers.append(
            {
                "name": "ALORA",
                "label": f"ALORA ({args.alora_port})",
                "model_path": args.alora_model,
                "port": args.alora_port,
            }
        )
    if args.server_type in ["lora", "both"]:
        servers.append(
            {
                "name": "LORA",
                "label": f"LORA ({args.lora_port})",
                "model_path": args.lora_model,
                "port": args.lora_port,
            }
        )

    # Clean up any stale processes
    if not args.skip_server_management:
        kill_stale_vllm_processes()

    # Run experiments
    for server_config in servers:
        server_name = server_config["name"]
        server_label = server_config["label"]
        model_path = server_config["model_path"]
        port = server_config["port"]

        print("\n" + "=" * 70)
        print(f"RUNNING EXPERIMENTS FOR {server_name}")
        print("=" * 70)

        # Create server-specific output directory
        server_output_dir = args.output_dir / server_name.lower()
        server_output_dir.mkdir(parents=True, exist_ok=True)

        result_files = []
        server = None

        for seed in range(1, args.seeds + 1):
            print(f"\n{'─' * 70}")
            print(f"SEED {seed} / {args.seeds}")
            print(f"{'─' * 70}")

            # Start server if needed
            if not args.skip_server_management and (
                server is None or not args.keep_servers_running
            ):
                log_file = server_output_dir / f"vllm_{server_name.lower()}_seed{seed}.log"
                server = VLLMServer(model_path, port, log_file, args.max_model_len)
                server.start(dry_run=args.dry_run)

            # Run benchmark
            try:
                result_file = run_benchmark(
                    benchmark_script=args.benchmark_script,
                    race_dir=args.race_dir,
                    server_name=server_label,
                    server_port=port,
                    model_path=model_path,
                    seed=seed,
                    runs=args.runs,
                    concurrency=args.concurrency,
                    top_k=args.top_k,
                    output_dir=server_output_dir,
                    dry_run=args.dry_run,
                )
                result_files.append(result_file)

            except Exception as e:
                print(f"[ERROR] Failed to run benchmark for seed {seed}: {e}")
                if not args.keep_servers_running and server:
                    server.stop(dry_run=args.dry_run)
                raise

            # Stop server if not keeping it running
            if not args.keep_servers_running and server:
                server.stop(dry_run=args.dry_run)
                server = None

        # Stop server if it's still running
        if server:
            server.stop(dry_run=args.dry_run)

        # Aggregate results
        if not args.dry_run and result_files:
            aggregate_path = server_output_dir / "aggregate_statistics.json"
            aggregate_results(result_files, aggregate_path)

            # Generate HTML reports
            generate_html_reports(
                result_files, args.race_dir, server_output_dir, server_label
            )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
