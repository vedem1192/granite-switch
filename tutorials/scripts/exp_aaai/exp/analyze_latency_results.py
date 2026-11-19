#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze latency experiment results and extract clean statistics.

This script properly handles the mixed server data from bench_pipeline_race.py
and extracts separate statistics for aLoRA and LoRA.

Usage:
    python analyze_latency_results.py latency_results/
    python analyze_latency_results.py latency_results/ --output-dir paper_stats/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing experiment results (with alora/ and lora/ subdirs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for cleaned results (default: results_dir/analysis/)",
    )
    return parser.parse_args()


def extract_server_data(mixed_results: dict, server_name: str) -> dict:
    """Extract data for one server from the mixed results."""
    if "servers" in mixed_results and server_name in mixed_results["servers"]:
        return mixed_results["servers"][server_name]
    return {}


def compute_step_statistics(conversations: list) -> dict:
    """Compute per-step latency statistics across all conversations."""
    step_latencies = {}

    for conv in conversations:
        for turn in conv.get("turns", []):
            for step in turn.get("steps", []):
                step_name = step.get("step", "unknown")
                duration = step.get("duration_s", 0)

                if step_name not in step_latencies:
                    step_latencies[step_name] = []
                step_latencies[step_name].append(duration)

    # Compute statistics per step
    step_stats = {}
    for step_name, latencies in step_latencies.items():
        step_stats[step_name] = {
            "count": len(latencies),
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        }

    return step_stats


def analyze_seed(result_file: Path, server_name: str) -> dict:
    """Analyze one seed for one server."""
    with open(result_file) as f:
        data = json.load(f)

    server_data = extract_server_data(data, server_name)

    if not server_data:
        print(f"  WARNING: No data for {server_name} in {result_file}")
        return {}

    conversations = server_data.get("conversations", [])
    wall_total = server_data.get("wall_total", 0)
    conv_walls = server_data.get("conv_walls", [])

    step_stats = compute_step_statistics(conversations)

    return {
        "total_conversations": len(conversations),
        "wall_total_s": wall_total,
        "wall_per_conversation_mean": float(np.mean(conv_walls)) if conv_walls else 0,
        "wall_per_conversation_median": float(np.median(conv_walls))
        if conv_walls
        else 0,
        "step_statistics": step_stats,
    }


def aggregate_across_seeds(seed_results: list[dict]) -> dict:
    """Compute mean and std across seeds."""
    if not seed_results:
        return {}

    # Aggregate scalar metrics
    aggregate = {
        "num_seeds": len(seed_results),
        "seeds_data": seed_results,
    }

    # Compute mean/std for scalar metrics
    scalar_keys = [
        "total_conversations",
        "wall_total_s",
        "wall_per_conversation_mean",
        "wall_per_conversation_median",
    ]

    for key in scalar_keys:
        values = [s[key] for s in seed_results if key in s]
        if values:
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_std"] = float(np.std(values))
            aggregate[f"{key}_min"] = float(np.min(values))
            aggregate[f"{key}_max"] = float(np.max(values))

    # Aggregate step statistics
    all_step_names = set()
    for seed in seed_results:
        all_step_names.update(seed.get("step_statistics", {}).keys())

    step_aggregates = {}
    for step_name in all_step_names:
        step_data = {}

        # Get each metric across seeds
        for metric in ["mean", "median", "std", "p95"]:
            values = []
            for seed in seed_results:
                step_stats = seed.get("step_statistics", {}).get(step_name, {})
                if metric in step_stats:
                    values.append(step_stats[metric])

            if values:
                step_data[f"{metric}_mean"] = float(np.mean(values))
                step_data[f"{metric}_std"] = float(np.std(values))

        step_aggregates[step_name] = step_data

    aggregate["step_aggregates"] = step_aggregates

    return aggregate


def print_summary(alora_stats: dict, lora_stats: dict):
    """Print a human-readable summary."""
    print("\n" + "=" * 80)
    print("LATENCY EXPERIMENT SUMMARY")
    print("=" * 80)

    if "wall_total_s_mean" in alora_stats and "wall_total_s_mean" in lora_stats:
        alora_time = alora_stats["wall_total_s_mean"]
        alora_std = alora_stats["wall_total_s_std"]
        lora_time = lora_stats["wall_total_s_mean"]
        lora_std = lora_stats["wall_total_s_std"]
        speedup = lora_time / alora_time if alora_time > 0 else 0

        print(f"\n📊 Overall Performance (mean ± std across {alora_stats['num_seeds']} seeds)")
        print(f"{'─' * 80}")
        print(f"  aLoRA:   {alora_time:6.1f} ± {alora_std:5.1f} seconds")
        print(f"  LoRA:    {lora_time:6.1f} ± {lora_std:5.1f} seconds")
        print(f"  Speedup: {speedup:.2f}× faster")

        if "total_conversations_mean" in alora_stats:
            n_conv = alora_stats["total_conversations_mean"]
            print(f"\n  Total conversations: {n_conv:.0f}")
            print(
                f"  Time per conversation (aLoRA): {alora_stats['wall_per_conversation_mean_mean']:.1f}s"
            )
            print(
                f"  Time per conversation (LoRA):  {lora_stats['wall_per_conversation_mean_mean']:.1f}s"
            )

    if "step_aggregates" in alora_stats and "step_aggregates" in lora_stats:
        print(f"\n📈 Per-Step Median Latencies (mean ± std across seeds)")
        print(f"{'─' * 80}")
        print(f"{'Step':<20} {'aLoRA (s)':<15} {'LoRA (s)':<15} {'Speedup':<10}")
        print(f"{'─' * 80}")

        all_steps = set(alora_stats["step_aggregates"].keys()) | set(
            lora_stats["step_aggregates"].keys()
        )

        for step_name in sorted(all_steps):
            alora_step = alora_stats["step_aggregates"].get(step_name, {})
            lora_step = lora_stats["step_aggregates"].get(step_name, {})

            alora_med = alora_step.get("median_mean", 0)
            alora_std_med = alora_step.get("median_std", 0)
            lora_med = lora_step.get("median_mean", 0)
            lora_std_med = lora_step.get("median_std", 0)

            speedup = lora_med / alora_med if alora_med > 0 else 0

            print(
                f"{step_name:<20} "
                f"{alora_med:5.3f}±{alora_std_med:5.3f}   "
                f"{lora_med:5.3f}±{lora_std_med:5.3f}   "
                f"{speedup:>5.2f}×"
            )

    print("=" * 80)


def main():
    args = parse_args()

    if not args.results_dir.exists():
        sys.exit(f"ERROR: Results directory not found: {args.results_dir}")

    output_dir = args.output_dir or (args.results_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing results from: {args.results_dir}")
    print(f"Output directory: {output_dir}")

    # Find all seed files (they're duplicated in both alora/ and lora/ dirs, so just use one)
    alora_dir = args.results_dir / "alora"
    if not alora_dir.exists():
        sys.exit(f"ERROR: alora/ subdirectory not found in {args.results_dir}")

    seed_files = sorted(alora_dir.glob("race_results_seed*.json"))

    if not seed_files:
        sys.exit(f"ERROR: No race_results_seed*.json files found in {alora_dir}")

    print(f"Found {len(seed_files)} seeds")

    # Analyze each seed for both servers
    alora_seeds = []
    lora_seeds = []

    for seed_file in seed_files:
        seed_num = seed_file.stem.replace("race_results_seed", "")
        print(f"\n[Seed {seed_num}]")

        alora_data = analyze_seed(seed_file, "ALORA (8111)")
        lora_data = analyze_seed(seed_file, "LORA (8112)")

        if alora_data:
            alora_seeds.append({"seed": int(seed_num), **alora_data})
            print(
                f"  aLoRA: {alora_data['total_conversations']} conversations, "
                f"{alora_data['wall_total_s']:.1f}s total"
            )

        if lora_data:
            lora_seeds.append({"seed": int(seed_num), **lora_data})
            print(
                f"  LoRA:  {lora_data['total_conversations']} conversations, "
                f"{lora_data['wall_total_s']:.1f}s total"
            )

    # Aggregate across seeds
    print("\n[Aggregating across seeds...]")
    alora_aggregate = aggregate_across_seeds(alora_seeds)
    lora_aggregate = aggregate_across_seeds(lora_seeds)

    # Save results
    alora_output = output_dir / "alora_statistics.json"
    lora_output = output_dir / "lora_statistics.json"
    comparison_output = output_dir / "comparison.json"

    with open(alora_output, "w") as f:
        json.dump(alora_aggregate, f, indent=2)
    print(f"\n[SAVED] aLoRA statistics: {alora_output}")

    with open(lora_output, "w") as f:
        json.dump(lora_aggregate, f, indent=2)
    print(f"[SAVED] LoRA statistics: {lora_output}")

    comparison = {
        "alora": alora_aggregate,
        "lora": lora_aggregate,
        "speedup": (
            lora_aggregate.get("wall_total_s_mean", 0)
            / alora_aggregate.get("wall_total_s_mean", 1)
        ),
    }

    with open(comparison_output, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[SAVED] Comparison: {comparison_output}")

    # Print summary
    print_summary(alora_aggregate, lora_aggregate)


if __name__ == "__main__":
    main()
