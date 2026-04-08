#!/usr/bin/env python3
"""
Analyze cost from baseline results.
Usage: python analyze_cost.py --result-dir baseline_results/cot_gpt5.4nano
"""

import argparse
import json
from pathlib import Path


def calculate_cost(usage: dict, reasoning_model: str, scoring_model: str) -> float:
    """
    Calculate cost in USD based on token usage and model pricing.
    
    Pricing (per 1M tokens):
    - GPT-5.4-nano: $0.20 input, $1.25 output
    - GPT-5: $1.25 input, $10 output (estimated)
    """
    PRICING = {
        "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
        "gpt-5": {"input": 1.25, "output": 10},  # Estimated, adjust as needed
        "default": {"input": 0.20, "output": 1.25}
    }
    
    def get_price(model_name: str):
        model_lower = model_name.lower()
        for key in PRICING:
            if key in model_lower:
                return PRICING[key]
        return PRICING["default"]
    
    reasoning_price = get_price(reasoning_model)
    scoring_price = get_price(scoring_model)
    
    reasoning_cost = (
        usage.get("reasoning_input", 0) * reasoning_price["input"] / 1_000_000 +
        usage.get("reasoning_output", 0) * reasoning_price["output"] / 1_000_000
    )
    scoring_cost = (
        usage.get("scoring_input", 0) * scoring_price["input"] / 1_000_000 +
        usage.get("scoring_output", 0) * scoring_price["output"] / 1_000_000
    )
    
    return reasoning_cost + scoring_cost


def main():
    parser = argparse.ArgumentParser(description='Analyze cost from baseline results')
    parser.add_argument('--result-dir', type=str, required=True,
                       help='Path to result directory containing results.json')
    parser.add_argument('--reasoning-model', type=str, default='gpt-5.4-nano',
                       help='Reasoning model name for pricing')
    parser.add_argument('--scoring-model', type=str, default='gpt-5',
                       help='Scoring model name for pricing')
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    results_path = result_dir / 'results.json'
    
    if not results_path.exists():
        print(f"Error: results.json not found in {result_dir}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"Analyzing {len(results)} users from {result_dir}")
    
    total_usage = {"reasoning_input": 0, "reasoning_output": 0, "scoring_input": 0, "scoring_output": 0}
    per_user_costs = []
    users_with_usage = 0
    
    for user_result in results:
        user_id = user_result['user_id']
        user_usage = {"reasoning_input": 0, "reasoning_output": 0, "scoring_input": 0, "scoring_output": 0}
        
        for turn_result in user_result.get('turn_results', []):
            if 'usage' in turn_result:
                users_with_usage += 1
                for key in user_usage:
                    user_usage[key] += turn_result['usage'].get(key, 0)
        
        if any(user_usage.values()):
            user_cost = calculate_cost(user_usage, args.reasoning_model, args.scoring_model)
            per_user_costs.append({
                "user_id": user_id,
                "usage": user_usage,
                "cost_usd": user_cost
            })
            
            for key in total_usage:
                total_usage[key] += user_usage[key]
    
    if not any(total_usage.values()):
        print("\n⚠️  No usage information found in results.")
        print("This might be an older run without token tracking.")
        print("Please re-run the baseline with the updated code to track costs.")
        return
    
    total_cost = calculate_cost(total_usage, args.reasoning_model, args.scoring_model)
    avg_cost = total_cost / len(results) if results else 0
    
    cost_report = {
        "total_usage": total_usage,
        "total_cost_usd": total_cost,
        "average_cost_per_user_usd": avg_cost,
        "n_users": len(results),
        "per_user_costs": per_user_costs[:10],  # First 10 users as sample
        "pricing_info": {
            "reasoning_model": args.reasoning_model,
            "scoring_model": args.scoring_model,
            "note": "GPT-5.4-nano: $0.20/1M input, $1.25/1M output tokens"
        }
    }
    
    # Save report
    report_path = result_dir / 'cost_report.json'
    with open(report_path, 'w') as f:
        json.dump(cost_report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("COST SUMMARY")
    print("="*60)
    print(f"Reasoning Model: {args.reasoning_model}")
    print(f"Scoring Model:   {args.scoring_model}")
    print(f"\nTotal Users:     {len(results)}")
    print(f"Users with data: {len(per_user_costs)}")
    print(f"\nToken Usage:")
    print(f"  Reasoning Input:  {total_usage['reasoning_input']:>12,} tokens")
    print(f"  Reasoning Output: {total_usage['reasoning_output']:>12,} tokens")
    print(f"  Scoring Input:    {total_usage['scoring_input']:>12,} tokens")
    print(f"  Scoring Output:   {total_usage['scoring_output']:>12,} tokens")
    print(f"  Total:            {sum(total_usage.values()):>12,} tokens")
    print(f"\nCost Breakdown:")
    
    reasoning_input_cost = total_usage['reasoning_input'] * 0.20 / 1_000_000
    reasoning_output_cost = total_usage['reasoning_output'] * 1.25 / 1_000_000
    scoring_input_cost = total_usage['scoring_input'] * 0.50 / 1_000_000  # Adjust if different
    scoring_output_cost = total_usage['scoring_output'] * 2.50 / 1_000_000  # Adjust if different
    
    print(f"  Reasoning Input:  ${reasoning_input_cost:>10.4f}")
    print(f"  Reasoning Output: ${reasoning_output_cost:>10.4f}")
    print(f"  Scoring Input:    ${scoring_input_cost:>10.4f}")
    print(f"  Scoring Output:   ${scoring_output_cost:>10.4f}")
    print(f"  {'─'*40}")
    print(f"  Total Cost:       ${total_cost:>10.4f} USD")
    print(f"\nAverage per user:   ${avg_cost:>10.4f} USD")
    print(f"\nCost report saved to: {report_path}")
    print("="*60)


if __name__ == '__main__':
    main()
