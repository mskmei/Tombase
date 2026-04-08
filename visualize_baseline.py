#!/usr/bin/env python3
"""
Independent visualization script for baseline results.
Usage: python visualize_baseline.py --result-dir baseline_results/cot_gpt5.4nano
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean
import matplotlib.pyplot as plt


def load_results(result_dir: Path):
    """Load results.json from the result directory."""
    results_path = result_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {result_dir}")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def aggregate_per_turn(results):
    """
    Aggregate per-turn metrics across all users.
    Returns dict with turn_index -> {metric_name -> list of values}
    """
    per_turn_data = defaultdict(lambda: {
        'accuracy': [],
        'ranking_score': [],
        'generation_score': [],
        'relative_gpt_score': [],
        'similarity_score': [],
        'relative_similarity_score': []
    })
    
    for user_result in results:
        user_id = user_result['user_id']
        for turn_result in user_result.get('turn_results', []):
            turn_idx = turn_result['turn']
            
            # Debug: print first few turns
            if turn_idx < 5:
                print(f"User {user_id[:8]}, Turn {turn_idx}: acc={turn_result.get('accuracy', 'N/A')}")
            
            per_turn_data[turn_idx]['accuracy'].append(turn_result.get('accuracy', 0.0))
            per_turn_data[turn_idx]['ranking_score'].append(turn_result.get('ranking_score', 0.0))
            per_turn_data[turn_idx]['generation_score'].append(turn_result.get('generation_score', 0.0))
            per_turn_data[turn_idx]['relative_gpt_score'].append(turn_result.get('relative_gpt_score', 0.0))
            per_turn_data[turn_idx]['similarity_score'].append(turn_result.get('similarity_score', 0.0))
            per_turn_data[turn_idx]['relative_similarity_score'].append(turn_result.get('relative_similarity_score', 0.0))
    
    return per_turn_data


def compute_statistics(per_turn_data):
    """Compute mean and count for each turn."""
    stats = {}
    
    print("\n=== Per-Turn Statistics ===")
    for turn_idx in sorted(per_turn_data.keys()):
        data = per_turn_data[turn_idx]
        
        stats[turn_idx] = {
            'accuracy': mean(data['accuracy']) if data['accuracy'] else 0.0,
            'ranking_score': mean(data['ranking_score']) if data['ranking_score'] else 0.0,
            'generation_score': mean(data['generation_score']) if data['generation_score'] else 0.0,
            'relative_gpt_score': mean(data['relative_gpt_score']) if data['relative_gpt_score'] else 0.0,
            'similarity_score': mean(data['similarity_score']) if data['similarity_score'] else 0.0,
            'relative_similarity_score': mean(data['relative_similarity_score']) if data['relative_similarity_score'] else 0.0,
            'count': len(data['accuracy'])
        }
        
        # Debug print for first few turns
        if turn_idx < 10:
            acc_values = data['accuracy']
            print(f"Turn {turn_idx}: accuracy values = {acc_values}, mean = {stats[turn_idx]['accuracy']:.3f}, count = {stats[turn_idx]['count']}")
    
    return stats


def plot_metrics(stats, output_path):
    """Plot 6 metrics over turns."""
    turns = sorted(stats.keys())
    if not turns:
        print("No turns to plot!")
        return
    
    x = [t + 1 for t in turns]  # 1-indexed for display
    acc = [stats[t]['accuracy'] for t in turns]
    rank = [stats[t]['ranking_score'] for t in turns]
    gen = [stats[t]['generation_score'] for t in turns]
    rel_gpt = [stats[t]['relative_gpt_score'] for t in turns]
    sim = [stats[t]['similarity_score'] for t in turns]
    rel_sim = [stats[t]['relative_similarity_score'] for t in turns]
    counts = [stats[t]['count'] for t in turns]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Accuracy
    axes[0, 0].plot(x, acc, marker='o', linewidth=2, color='tab:blue')
    axes[0, 0].set_title('Accuracy by Turn')
    axes[0, 0].set_xlabel('Turn')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].grid(True, alpha=0.3)
    # Annotate sample counts
    for i, (xi, yi, ci) in enumerate(zip(x, acc, counts)):
        if i < 15:  # Only annotate first 15 turns to avoid clutter
            axes[0, 0].annotate(f'n={ci}', (xi, yi), textcoords='offset points',
                               xytext=(0, 5), ha='center', fontsize=8, alpha=0.7)
    
    # Ranking Score
    axes[0, 1].plot(x, rank, marker='s', linewidth=2, color='tab:orange')
    axes[0, 1].set_title('Ranking Score by Turn')
    axes[0, 1].set_xlabel('Turn')
    axes[0, 1].set_ylabel('Ranking Score')
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Generation Score
    axes[0, 2].plot(x, gen, marker='^', linewidth=2, color='tab:green')
    axes[0, 2].set_title('Generation Score by Turn')
    axes[0, 2].set_xlabel('Turn')
    axes[0, 2].set_ylabel('GPT Score (0-5)')
    axes[0, 2].set_ylim(0.0, 5.0)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Relative GPT Score
    axes[1, 0].plot(x, rel_gpt, marker='d', linewidth=2, color='tab:red')
    axes[1, 0].set_title('Relative GPT Score by Turn')
    axes[1, 0].set_xlabel('Turn')
    axes[1, 0].set_ylabel('Relative GPT Score')
    axes[1, 0].set_ylim(-5.0, 5.0)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Similarity Score
    axes[1, 1].plot(x, sim, marker='p', linewidth=2, color='tab:purple')
    axes[1, 1].set_title('Similarity Score by Turn')
    axes[1, 1].set_xlabel('Turn')
    axes[1, 1].set_ylabel('Embedding Similarity')
    axes[1, 1].set_ylim(-1.0, 1.0)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Relative Similarity Score
    axes[1, 2].plot(x, rel_sim, marker='h', linewidth=2, color='tab:brown')
    axes[1, 2].set_title('Relative Similarity Score by Turn')
    axes[1, 2].set_xlabel('Turn')
    axes[1, 2].set_ylabel('Relative Similarity')
    axes[1, 2].set_ylim(-2.0, 2.0)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize baseline results independently')
    parser.add_argument('--result-dir', type=str, required=True,
                       help='Path to result directory containing results.json')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (default: result-dir/debug_plot.png)')
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")
    
    output_path = args.output if args.output else result_dir / 'debug_plot.png'
    
    print(f"Loading results from: {result_dir}")
    results = load_results(result_dir)
    print(f"Loaded {len(results)} users")
    
    print("\n=== Aggregating per-turn data ===")
    per_turn_data = aggregate_per_turn(results)
    
    print("\n=== Computing statistics ===")
    stats = compute_statistics(per_turn_data)
    
    print("\n=== Plotting ===")
    plot_metrics(stats, output_path)
    
    # Save debug stats to JSON
    debug_stats_path = result_dir / 'debug_stats.json'
    with open(debug_stats_path, 'w') as f:
        json.dump({str(k): v for k, v in stats.items()}, f, indent=2)
    print(f"Debug statistics saved to: {debug_stats_path}")
    
    print("\n=== Done ===")


if __name__ == '__main__':
    main()
