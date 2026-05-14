#!/usr/bin/env python3
"""
Visualization script for CoT baseline results (v2 format).

Results are stored in:
  {run_dir}/users/{user_id}.json   — per-user record (v2 format)
  {run_dir}/summary.json           — global summary (v2 format)

Usage:
    python visualize_baseline.py --result-dir baseline_results/cot_gpt5.4nano
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean
import matplotlib.pyplot as plt


def load_summary(result_dir: Path) -> dict:
    """Load summary.json (v2 format)."""
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {result_dir}")
    with open(summary_path) as f:
        return json.load(f)


def load_user_records(result_dir: Path) -> list:
    """Load all per-user JSON files from users/ subdirectory."""
    users_dir = result_dir / "users"
    if not users_dir.exists():
        raise FileNotFoundError(f"users/ directory not found in {result_dir}")
    records = []
    for p in sorted(users_dir.glob("*.json")):
        with open(p) as f:
            records.append(json.load(f))
    return records


def aggregate_per_turn(user_records: list) -> dict:
    """
    Aggregate per-turn metrics across all users.
    Returns dict: turn_index -> {metric -> list of values}
    """
    per_turn: dict = defaultdict(lambda: defaultdict(list))

    for rec in user_records:
        user_id = rec.get("user", "?")
        for turn in rec.get("turns", []):
            idx = turn["turn_index"]
            pred = turn.get("prediction", {})
            adapt = turn.get("adaptation", {})

            if pred.get("success"):
                per_turn[idx]["prediction_accuracy"].append(pred["accuracy"])
                per_turn[idx]["prediction_ranking_score"].append(pred["ranking_score"])

            if adapt.get("success"):
                for key in ["gpt_score", "relative_gpt_score", "relative_mean_gpt_score",
                            "similarity_score", "relative_score", "relative_mean_score"]:
                    val = adapt.get(key)
                    if val is not None:
                        per_turn[idx][key].append(val)

    return per_turn


def compute_stats(per_turn: dict) -> dict:
    """Compute mean per turn; return turn_index -> {metric -> mean}."""
    stats = {}
    print("\n=== Per-Turn Statistics (first 10 turns) ===")
    for idx in sorted(per_turn.keys()):
        data = per_turn[idx]
        stats[idx] = {k: mean(v) if v else 0.0 for k, v in data.items()}
        stats[idx]["n_prediction"] = len(data.get("prediction_accuracy", []))
        stats[idx]["n_adaptation"] = len(data.get("gpt_score", []))
        if idx < 10:
            print(
                f"Turn {idx}: acc={stats[idx].get('prediction_accuracy', 0):.3f}  "
                f"rank={stats[idx].get('prediction_ranking_score', 0):.3f}  "
                f"gpt={stats[idx].get('gpt_score', 0):.3f}  "
                f"rel_gpt={stats[idx].get('relative_gpt_score', 0):.3f}  "
                f"sim={stats[idx].get('similarity_score', 0):.3f}  "
                f"(n_pred={stats[idx]['n_prediction']}, n_adapt={stats[idx]['n_adaptation']})"
            )
    return stats


def plot_metrics(stats: dict, output_path):
    turns = sorted(stats.keys())
    if not turns:
        print("No turns to plot!")
        return

    x = [t + 1 for t in turns]

    def _vals(key):
        return [stats[t].get(key, 0.0) for t in turns]

    counts_pred = [stats[t].get("n_prediction", 0) for t in turns]
    counts_adapt = [stats[t].get("n_adaptation", 0) for t in turns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(x, _vals("prediction_accuracy"), marker="o", linewidth=2, color="tab:blue")
    axes[0, 0].set_title("Prediction Accuracy by Turn")
    axes[0, 0].set_xlabel("Turn")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].grid(True, alpha=0.3)
    for i, (xi, yi, ci) in enumerate(zip(x, _vals("prediction_accuracy"), counts_pred)):
        if i < 15:
            axes[0, 0].annotate(f"n={ci}", (xi, yi), textcoords="offset points",
                                xytext=(0, 5), ha="center", fontsize=8, alpha=0.7)

    axes[0, 1].plot(x, _vals("prediction_ranking_score"), marker="s", linewidth=2, color="tab:orange")
    axes[0, 1].set_title("Prediction Ranking Score by Turn")
    axes[0, 1].set_xlabel("Turn")
    axes[0, 1].set_ylabel("Ranking Score")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(x, _vals("gpt_score"), marker="^", linewidth=2, color="tab:green")
    axes[0, 2].set_title("GPT Score by Turn")
    axes[0, 2].set_xlabel("Turn")
    axes[0, 2].set_ylabel("GPT Score (0-5)")
    axes[0, 2].set_ylim(0.0, 5.0)
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(x, _vals("relative_gpt_score"), marker="d", linewidth=2, color="tab:red",
                    label="vs max")
    axes[1, 0].plot(x, _vals("relative_mean_gpt_score"), marker="x", linewidth=1.5,
                    color="tab:pink", linestyle="--", label="vs mean")
    axes[1, 0].set_title("Relative GPT Score by Turn")
    axes[1, 0].set_xlabel("Turn")
    axes[1, 0].set_ylabel("Relative GPT Score")
    axes[1, 0].set_ylim(-5.0, 5.0)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(x, _vals("similarity_score"), marker="p", linewidth=2, color="tab:purple")
    axes[1, 1].set_title("Embedding Similarity Score by Turn")
    axes[1, 1].set_xlabel("Turn")
    axes[1, 1].set_ylabel("Cosine Similarity")
    axes[1, 1].set_ylim(-1.0, 1.0)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(x, _vals("relative_score"), marker="h", linewidth=2, color="tab:brown",
                    label="vs max")
    axes[1, 2].plot(x, _vals("relative_mean_score"), marker="x", linewidth=1.5,
                    color="tab:olive", linestyle="--", label="vs mean")
    axes[1, 2].set_title("Relative Similarity Score by Turn")
    axes[1, 2].set_xlabel("Turn")
    axes[1, 2].set_ylabel("Relative Similarity")
    axes[1, 2].set_ylim(-2.0, 2.0)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize CoT baseline results (v2 format)")
    parser.add_argument("--result-dir", type=str, required=True,
                        help="Path to result directory (contains summary.json and users/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for plot (default: result-dir/turn_trends.png)")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    output_path = args.output or result_dir / "turn_trends.png"

    print(f"Loading results from: {result_dir}")
    user_records = load_user_records(result_dir)
    print(f"Loaded {len(user_records)} user records")

    # Also load and print summary if available
    try:
        summary = load_summary(result_dir)
        overall_pred = summary.get("overall_prediction", {})
        overall_adapt = summary.get("overall_adaptation", {})
        print("\n=== Overall Summary ===")
        for k, v in {**overall_pred, **overall_adapt}.items():
            print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: None")
    except FileNotFoundError:
        summary = None

    print("\n=== Aggregating per-turn data ===")
    per_turn = aggregate_per_turn(user_records)

    print("\n=== Computing statistics ===")
    stats = compute_stats(per_turn)

    print("\n=== Plotting ===")
    plot_metrics(stats, output_path)

    debug_stats_path = result_dir / "debug_stats.json"
    with open(debug_stats_path, "w") as f:
        json.dump({str(k): v for k, v in stats.items()}, f, indent=2)
    print(f"Debug statistics saved to: {debug_stats_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

