#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge separate aLoRA and LoRA results and generate race HTML visualizations.

This script takes the clean separated results from run_latency_experiments.py
and merges them back into the 2-server race format needed for HTML visualizations.

Usage:
    python generate_race_html.py latency_results/
    python generate_race_html.py latency_results/ --output-dir race_visualizations/
    python generate_race_html.py latency_results/ --seed 1  # Generate only seed 1
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing clean results (with alora/ and lora/ subdirs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for HTML files (default: results_dir/visualizations/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Generate HTML for specific seed only (default: all seeds)",
    )
    parser.add_argument(
        "--race-templates",
        type=Path,
        default=Path("/data/vdemers/granite-switch-benchmark/granite-switch/tutorials/scripts/comparison/alora_vs_lora_race/sample_run"),
        help="Directory containing HTML templates",
    )
    return parser.parse_args()


def merge_results(alora_result: dict, lora_result: dict) -> dict:
    """Merge separate aLoRA and LoRA results into 2-server race format."""
    # Create the mixed format that the HTML templates expect
    merged = {
        "metadata": alora_result.get("metadata", {}),
        "servers": {
            alora_result.get("server_name", "ALORA (8111)"): {
                "model": alora_result.get("model", ""),
                "adapter_tech": alora_result.get("adapter_tech", {}),
                "wall_total": alora_result.get("wall_total", 0),
                "conv_walls": alora_result.get("conv_walls", []),
                "conversations": alora_result.get("conversations", []),
                "exits": alora_result.get("exits", {}),
                "step_medians": alora_result.get("step_medians", {}),
            },
            lora_result.get("server_name", "LORA (8112)"): {
                "model": lora_result.get("model", ""),
                "adapter_tech": lora_result.get("adapter_tech", {}),
                "wall_total": lora_result.get("wall_total", 0),
                "conv_walls": lora_result.get("conv_walls", []),
                "conversations": lora_result.get("conversations", []),
                "exits": lora_result.get("exits", {}),
                "step_medians": lora_result.get("step_medians", {}),
            },
        },
    }

    return merged


def merge_events(alora_events: dict, lora_events: dict) -> dict:
    """Merge separate aLoRA and LoRA events into 2-server race format."""
    # Combine events from both servers and sort by timestamp
    alora_event_list = alora_events.get("events", [])
    lora_event_list = lora_events.get("events", [])

    all_events = alora_event_list + lora_event_list
    all_events.sort(key=lambda e: e.get("t", 0))

    # Calculate race wall time (max of both servers)
    alora_max_t = max((e.get("t", 0) for e in alora_event_list), default=0)
    lora_max_t = max((e.get("t", 0) for e in lora_event_list), default=0)
    race_wall = max(alora_max_t, lora_max_t)

    alora_server = alora_events.get("server_name", "ALORA (8111)")
    lora_server = lora_events.get("server_name", "LORA (8112)")

    merged = {
        "metadata": alora_events.get("metadata", {}),
        "race_wall": race_wall,
        "servers": [alora_server, lora_server],
        "events": all_events,
    }

    # Update metadata race_wall if present
    if "metadata" in merged:
        merged["metadata"]["race_wall"] = race_wall

    return merged


def embed_json_in_html(
    template_path: Path,
    json_data: dict,
    output_path: Path,
    var_name: str,
    marker: str,
):
    """Embed JSON data into an HTML template."""
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path) as f:
        html = f.read()

    # Replace the marker with actual data
    html = re.sub(
        rf"const {var_name} = .*?; // {re.escape(marker)}",
        f"const {var_name} = {json.dumps(json_data)}; // {marker}",
        html,
        flags=re.DOTALL,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def compute_aggregate_statistics(merged_results_files: list[Path]) -> dict:
    """Compute aggregate statistics across all seeds."""
    import numpy as np

    all_seeds = []

    for seed_file in merged_results_files:
        with open(seed_file) as f:
            data = json.load(f)

        # Extract data for both servers
        alora_data = data["servers"].get("ALORA (8111)", {})
        lora_data = data["servers"].get("LORA (8112)", {})

        seed_stats = {
            "alora": {
                "wall_total": alora_data.get("wall_total", 0),
                "num_conversations": len(alora_data.get("conversations", [])),
            },
            "lora": {
                "wall_total": lora_data.get("wall_total", 0),
                "num_conversations": len(lora_data.get("conversations", [])),
            }
        }

        # Extract per-step medians
        alora_step_medians = alora_data.get("step_medians", {})
        lora_step_medians = lora_data.get("step_medians", {})

        seed_stats["alora"]["step_medians"] = alora_step_medians
        seed_stats["lora"]["step_medians"] = lora_step_medians

        all_seeds.append(seed_stats)

    if not all_seeds:
        return {}

    # Compute aggregates
    alora_times = [s["alora"]["wall_total"] for s in all_seeds]
    lora_times = [s["lora"]["wall_total"] for s in all_seeds]

    aggregate = {
        "num_seeds": len(all_seeds),
        "alora": {
            "wall_total_mean": float(np.mean(alora_times)),
            "wall_total_std": float(np.std(alora_times)),
            "wall_total_min": float(np.min(alora_times)),
            "wall_total_max": float(np.max(alora_times)),
        },
        "lora": {
            "wall_total_mean": float(np.mean(lora_times)),
            "wall_total_std": float(np.std(lora_times)),
            "wall_total_min": float(np.min(lora_times)),
            "wall_total_max": float(np.max(lora_times)),
        },
        "speedup": float(np.mean(lora_times) / np.mean(alora_times)) if np.mean(alora_times) > 0 else 0,
    }

    # Aggregate step medians
    all_step_names = set()
    for seed in all_seeds:
        all_step_names.update(seed["alora"]["step_medians"].keys())
        all_step_names.update(seed["lora"]["step_medians"].keys())

    alora_steps = {}
    lora_steps = {}

    for step_name in all_step_names:
        alora_values = [s["alora"]["step_medians"].get(step_name, 0) for s in all_seeds]
        lora_values = [s["lora"]["step_medians"].get(step_name, 0) for s in all_seeds]

        alora_steps[step_name] = {
            "mean": float(np.mean(alora_values)),
            "std": float(np.std(alora_values)),
        }
        lora_steps[step_name] = {
            "mean": float(np.mean(lora_values)),
            "std": float(np.std(lora_values)),
        }

    aggregate["alora"]["step_medians"] = alora_steps
    aggregate["lora"]["step_medians"] = lora_steps

    return aggregate


def generate_html_for_seed(
    seed_num: int,
    alora_dir: Path,
    lora_dir: Path,
    output_dir: Path,
    templates_dir: Path,
):
    """Generate HTML visualizations for one seed."""
    print(f"\n[Seed {seed_num}]")

    # Load separate results
    alora_results_file = alora_dir / f"race_results_seed{seed_num}.json"
    lora_results_file = lora_dir / f"race_results_seed{seed_num}.json"
    alora_events_file = alora_dir / f"race_events_seed{seed_num}.json"
    lora_events_file = lora_dir / f"race_events_seed{seed_num}.json"

    if not alora_results_file.exists():
        print(f"  [SKIP] aLoRA results not found: {alora_results_file}")
        return

    if not lora_results_file.exists():
        print(f"  [SKIP] LoRA results not found: {lora_results_file}")
        return

    with open(alora_results_file) as f:
        alora_results = json.load(f)

    with open(lora_results_file) as f:
        lora_results = json.load(f)

    with open(alora_events_file) as f:
        alora_events = json.load(f)

    with open(lora_events_file) as f:
        lora_events = json.load(f)

    # Merge into race format
    print(f"  [MERGE] Combining aLoRA and LoRA data...")
    merged_results = merge_results(alora_results, lora_results)
    merged_events = merge_events(alora_events, lora_events)

    # Save merged JSON files
    merged_results_file = output_dir / f"race_results_seed{seed_num}.json"
    merged_events_file = output_dir / f"race_events_seed{seed_num}.json"

    with open(merged_results_file, "w") as f:
        json.dump(merged_results, f, indent=2)
    print(f"  [SAVED] Merged results: {merged_results_file}")

    with open(merged_events_file, "w") as f:
        json.dump(merged_events, f, indent=2)
    print(f"  [SAVED] Merged events: {merged_events_file}")

    # Generate HTML files
    live_template = templates_dir / "race_live.html"
    report_template = templates_dir / "race_report.html"

    if not live_template.exists() or not report_template.exists():
        print(f"  [WARNING] HTML templates not found in {templates_dir}")
        print(f"            Skipping HTML generation")
        return

    live_html = output_dir / f"race_live_seed{seed_num}.html"
    report_html = output_dir / f"race_report_seed{seed_num}.html"

    print(f"  [HTML] Generating visualizations...")
    embed_json_in_html(
        live_template,
        merged_events,
        live_html,
        "RACE_EVENTS_EMBEDDED",
        "<<RACE_EVENTS>>",
    )
    print(f"  [SAVED] Live race: {live_html}")

    embed_json_in_html(
        report_template,
        merged_results,
        report_html,
        "RACE_DATA_EMBEDDED",
        "<<RACE_DATA>>",
    )
    print(f"  [SAVED] Report: {report_html}")

    # Print summary statistics
    alora_time = merged_results["servers"][alora_results.get("server_name", "ALORA (8111)")]["wall_total"]
    lora_time = merged_results["servers"][lora_results.get("server_name", "LORA (8112)")]["wall_total"]
    speedup = lora_time / alora_time if alora_time > 0 else 0

    print(f"  [STATS] aLoRA: {alora_time:.1f}s, LoRA: {lora_time:.1f}s, Speedup: {speedup:.2f}×")


def main():
    args = parse_args()

    if not args.results_dir.exists():
        sys.exit(f"ERROR: Results directory not found: {args.results_dir}")

    alora_dir = args.results_dir / "alora"
    lora_dir = args.results_dir / "lora"

    if not alora_dir.exists():
        sys.exit(f"ERROR: aLoRA directory not found: {alora_dir}")

    if not lora_dir.exists():
        sys.exit(f"ERROR: LoRA directory not found: {lora_dir}")

    output_dir = args.output_dir or (args.results_dir / "visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    templates_dir = args.race_templates
    if not templates_dir.exists():
        print(f"WARNING: Templates directory not found: {templates_dir}")
        print(f"         Will save merged JSON but skip HTML generation")

    print(f"Results directory: {args.results_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Templates:         {templates_dir}")

    # Find all seeds
    alora_seed_files = sorted(alora_dir.glob("race_results_seed*.json"))
    seed_numbers = []
    for f in alora_seed_files:
        match = re.match(r"race_results_seed(\d+)\.json", f.name)
        if match:
            seed_numbers.append(int(match.group(1)))

    if not seed_numbers:
        sys.exit(f"ERROR: No seed files found in {alora_dir}")

    # Filter by --seed if specified
    if args.seed is not None:
        if args.seed in seed_numbers:
            seed_numbers = [args.seed]
        else:
            sys.exit(f"ERROR: Seed {args.seed} not found. Available: {seed_numbers}")

    print(f"\nGenerating visualizations for {len(seed_numbers)} seed(s): {seed_numbers}")
    print("=" * 80)

    # Generate HTML for each seed
    merged_result_files = []
    for seed_num in seed_numbers:
        try:
            generate_html_for_seed(
                seed_num, alora_dir, lora_dir, output_dir, templates_dir
            )
            # Track merged results for aggregation
            merged_file = output_dir / f"race_results_seed{seed_num}.json"
            if merged_file.exists():
                merged_result_files.append(merged_file)
        except Exception as e:
            print(f"  [ERROR] Failed to generate HTML for seed {seed_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute aggregate statistics
    if merged_result_files:
        print("\n" + "=" * 80)
        print("COMPUTING AGGREGATE STATISTICS")
        print("=" * 80)

        aggregate_stats = compute_aggregate_statistics(merged_result_files)

        if aggregate_stats:
            # Save aggregate statistics
            aggregate_file = output_dir / "aggregate_statistics.json"
            with open(aggregate_file, "w") as f:
                json.dump(aggregate_stats, f, indent=2)
            print(f"[SAVED] {aggregate_file}")

            # Print summary
            print(f"\n📊 Summary (mean ± std across {aggregate_stats['num_seeds']} seeds)")
            print("─" * 80)
            alora_mean = aggregate_stats["alora"]["wall_total_mean"]
            alora_std = aggregate_stats["alora"]["wall_total_std"]
            lora_mean = aggregate_stats["lora"]["wall_total_mean"]
            lora_std = aggregate_stats["lora"]["wall_total_std"]
            speedup = aggregate_stats["speedup"]

            print(f"aLoRA:   {alora_mean:6.1f} ± {alora_std:5.1f}s")
            print(f"LoRA:    {lora_mean:6.1f} ± {lora_std:5.1f}s")
            print(f"Speedup: {speedup:.2f}×")

            # Print per-step stats
            print(f"\nPer-step latencies (mean ± std):")
            alora_steps = aggregate_stats["alora"]["step_medians"]
            lora_steps = aggregate_stats["lora"]["step_medians"]

            for step_name in sorted(alora_steps.keys()):
                alora_mean_step = alora_steps[step_name]["mean"]
                alora_std_step = alora_steps[step_name]["std"]
                lora_mean_step = lora_steps.get(step_name, {}).get("mean", 0)
                lora_std_step = lora_steps.get(step_name, {}).get("std", 0)
                step_speedup = lora_mean_step / alora_mean_step if alora_mean_step > 0 else 0

                print(f"  {step_name:20s}: "
                      f"aLoRA {alora_mean_step:.3f}±{alora_std_step:.3f}s, "
                      f"LoRA {lora_mean_step:.3f}±{lora_std_step:.3f}s, "
                      f"{step_speedup:.2f}×")

    print("\n" + "=" * 80)
    print("HTML GENERATION COMPLETE")
    print("=" * 80)
    print(f"Open visualizations: {output_dir}/race_live_seed*.html")
    print(f"Aggregate statistics: {output_dir}/aggregate_statistics.json")


if __name__ == "__main__":
    main()
