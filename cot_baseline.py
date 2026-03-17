import argparse
import json
import os
import random
import re
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Tuple

from data import Turn, load_data
from model import GenerationConfig, HFModel, OpenAIModel


def _plot_turn_trends(per_turn_stats: Dict[str, Dict[str, float]], output_path: str):
    import matplotlib.pyplot as plt

    turns = sorted(int(t) for t in per_turn_stats.keys())
    if not turns:
        return

    x = [t + 1 for t in turns]
    acc = [per_turn_stats[str(t)]["accuracy"] for t in turns]
    rank = [per_turn_stats[str(t)]["ranking_score"] for t in turns]
    gen = [per_turn_stats[str(t)]["generation_score"] for t in turns]
    rel = [per_turn_stats[str(t)]["relative_gpt_score"] for t in turns]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    axes[0].plot(x, acc, marker="o", linewidth=2, color="tab:blue")
    axes[0].set_title("Accuracy by Turn (Full-Length Users)")
    axes[0].set_xlabel("Turn")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, rank, marker="s", linewidth=2, color="tab:orange")
    axes[1].set_title("Ranking Score by Turn (Full-Length Users)")
    axes[1].set_xlabel("Turn")
    axes[1].set_ylabel("Ranking Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, gen, marker="^", linewidth=2, color="tab:green")
    axes[2].set_title("Generation Score by Turn (Full-Length Users)")
    axes[2].set_xlabel("Turn")
    axes[2].set_ylabel("Generation Score")
    axes[2].set_ylim(0.0, 5.0)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(x, rel, marker="d", linewidth=2, color="tab:red")
    axes[3].set_title("Relative GPT Score by Turn (Full-Length Users)")
    axes[3].set_xlabel("Turn")
    axes[3].set_ylabel("Relative GPT Score")
    axes[3].set_ylim(-5.0, 5.0)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


COT_RANK_PROMPT = """
You are selecting the best response for the current user message.

Given:
- Recent conversation context (up to {max_history_turns} previous turns)
- Current user message
- Candidate responses with ids

Task:
1. Reason internally step by step from concrete observable differences.
2. Rank all candidates from best to worst for this specific turn.

Rules:
- Use only evidence in the provided context and candidate texts.
- Do not speculate about hidden user traits.
- Do not output chain-of-thought.
- Output valid JSON only.

Return exactly:
{{
  "reason": "Brief 1-2 sentence summary of key differences.",
  "ranking": [1, 2, 3]
}}

[Recent conversation]
{history}

[Current user message]
{current_message}

[Candidates]
{candidates}
"""


EVALUATE_PROMPT = """
You are evaluating the similarity between an Adapted response and a set of candidate responses in the context of user preferences.

Inputs:
- Current turn with c candidates
- Adapted response

Step 1:
Identify 2-4 preference-relevant dimensions that distinguish the candidates.
Use short canonical labels (e.g., "values", "information_density", "structure", "actionability", "tone", "framing", "abstraction").

Step 2:
Using those dimensions, score how similar the Adapted response is to EACH candidate.
Scores are 0-5:
5 = Near-identical on key dimensions.
4 = Strong match on most key dimensions.
3 = Partial match.
2 = Weak match.
1 = Very weak match.
0 = Opposes/contradicts key dimensions.

Output valid JSON only:
{
    "dimensions": ["...", "..."],
    "scores": [s0, s1, ..., s{c-1}],
    "justification": "Brief explanation."
}

Rules:
- scores length MUST equal number of candidates.
- scores[i] corresponds to candidate i.
- Use the same dimensions for scoring all candidates.

[Current turn]
{current_turn}

[Number of candidates]
c={c}

[Adapted]
{adapted}
"""


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _compact_text(text: str, max_chars: int) -> str:
    text = _collapse_ws(text)
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]} ... {text[-half:]}"


def _build_history(conversation_history: List[Turn], max_history_turns: int, max_chars: int) -> str:
    prev = conversation_history[:-1][-max_history_turns:]
    if not prev:
        return "(empty)"
    lines = []
    for idx, t in enumerate(prev, start=1):
        lines.append(f"Turn-{idx} User: {_compact_text(t.user_message, max_chars)}")
        lines.append(f"Turn-{idx} Assistant(chosen): {_compact_text(t.chosen, max_chars)}")
    return "\n".join(lines)


def _build_candidates(turn: Turn, max_chars: int) -> str:
    lines = []
    for i, c in enumerate(turn.candidates, start=1):
        lines.append(f"{i}. {_compact_text(c, max_chars)}")
    return "\n".join(lines)


def _extract_json(text: str) -> Dict:
    # Remove visible reasoning tags if model emits them.
    cleaned = re.sub(r"<think>[\\s\\S]*?</think>", "", text, flags=re.IGNORECASE)
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]

    # Remove markdown code fences if present.
    cleaned = cleaned.replace("```json", "").replace("```", "")

    left = cleaned.find("{")
    right = cleaned.rfind("}")
    if left == -1 or right == -1 or right < left:
        raise ValueError(f"No JSON object found in output: {text}")
    return json.loads(cleaned[left : right + 1])


def _repair_ranking_output(
    reasoning_model,
    raw_output: str,
    n_candidates: int,
    reasoning_cfg: GenerationConfig,
) -> List[int]:
    repair_prompt = f"""
Convert the following model output into valid JSON only.

Requirements:
- Output exactly one JSON object
- Include key \"ranking\" as a full permutation of integers 1..{n_candidates}
- Include key \"reason\" as one short sentence
- No markdown, no extra text

Text to convert:
{raw_output}
"""
    repaired = reasoning_model.generate(repair_prompt, cfg=reasoning_cfg)["output"]
    repaired_json = _extract_json(repaired)
    return _normalize_ranking(repaired_json.get("ranking"), n_candidates=n_candidates)


def _normalize_ranking(raw_ranking, n_candidates: int) -> List[int]:
    if not isinstance(raw_ranking, list):
        raise ValueError("ranking must be a list")
    ranking: List[int] = []
    for item in raw_ranking:
        if isinstance(item, str):
            if item.strip().isdigit():
                ranking.append(int(item.strip()))
            else:
                raise ValueError(f"ranking item is not numeric: {item}")
        elif isinstance(item, int):
            ranking.append(item)
        else:
            raise ValueError(f"ranking item has unsupported type: {type(item)}")

    expected = list(range(1, n_candidates + 1))
    if sorted(ranking) != expected:
        raise ValueError(f"ranking must be a permutation of {expected}, got {ranking}")
    return ranking


def _resolve_chosen_idx(turn: Turn) -> int:
    """Return a valid chosen index when available, otherwise -1."""
    if 0 <= turn.chosen_idx < len(turn.candidates):
        return turn.chosen_idx
    if turn.chosen and turn.chosen in turn.candidates:
        return turn.candidates.index(turn.chosen)
    return -1


def build_model(
    backend: str,
    model_name: str,
    base_url: str,
    api_key: str,
    hf_enable_thinking: bool,
):
    if backend == "hf":
        return HFModel(model_name=model_name, enable_thinking=hf_enable_thinking)
    if backend == "openai":
        return OpenAIModel(api_key=api_key, base_url=base_url, model=model_name)
    raise ValueError(f"Unsupported backend: {backend}")


def predict_ranking_and_metrics(
    reasoning_model,
    scoring_model,
    conversation_history: List[Turn],
    reasoning_cfg: GenerationConfig,
    score_cfg: GenerationConfig,
    max_history_turns: int,
    max_chars: int,
    ranking_fail_mode: str = "fallback",
) -> Dict[str, float]:
    current_turn = conversation_history[-1]
    resolved_chosen_idx = _resolve_chosen_idx(current_turn)
    if resolved_chosen_idx < 0:
        raise ValueError("Current turn has no valid chosen candidate index.")

    history = _build_history(conversation_history, max_history_turns=max_history_turns, max_chars=max_chars)
    current_message = _compact_text(current_turn.user_message, max_chars)
    candidates = _build_candidates(current_turn, max_chars=max_chars)

    prompt = COT_RANK_PROMPT.format(
        max_history_turns=max_history_turns,
        history=history,
        current_message=current_message,
        candidates=candidates,
    )

    retries = 0
    n_candidates = len(current_turn.candidates)
    last_exc = None
    while True:
        try:
            output = reasoning_model.generate(prompt, cfg=reasoning_cfg)["output"]
            ranking_json = _extract_json(output)
            ranking = _normalize_ranking(ranking_json.get("ranking"), n_candidates=n_candidates)
            break
        except Exception as exc:
            last_exc = exc
            # One repair attempt from raw model text before consuming a retry.
            try:
                ranking = _repair_ranking_output(
                    reasoning_model=reasoning_model,
                    raw_output=output if "output" in locals() else str(exc),
                    n_candidates=n_candidates,
                    reasoning_cfg=reasoning_cfg,
                )
                break
            except Exception as repair_exc:
                last_exc = repair_exc
            retries += 1
            if retries > reasoning_cfg.max_retries:
                if ranking_fail_mode == "fallback":
                    print(
                        f"[WARN] Ranking parse failed after retries; fallback ranking used. Last error: {last_exc}"
                    )
                    ranking = list(range(1, n_candidates + 1))
                    break
                raise ValueError(
                    f"Failed to get valid ranking after {reasoning_cfg.max_retries} retries: {last_exc}"
                )

    gt_choice = resolved_chosen_idx + 1
    rank = ranking.index(gt_choice) + 1
    ranking_score = (len(ranking) - rank) / (len(ranking) - 1)
    accuracy = 1.0 if rank == 1 else 0.0

    top_choice_idx = ranking[0] - 1
    adapted_response = current_turn.candidates[top_choice_idx]
    eval_prompt = EVALUATE_PROMPT.format(
        current_turn=current_turn.format(include_candidates=True, include_choice=False),
        c=n_candidates,
        adapted=adapted_response,
    )

    eval_retries = 0
    while True:
        try:
            score_output = scoring_model.generate(eval_prompt, cfg=score_cfg)["output"]
            score_json = _extract_json(score_output)
            scores = score_json["scores"]
            if not isinstance(scores, list) or len(scores) != n_candidates:
                raise ValueError(f"scores length mismatch: expected {n_candidates}, got {len(scores) if isinstance(scores, list) else 'non-list'}")
            scores = [float(s) for s in scores]
            generation_score = scores[resolved_chosen_idx]
            if n_candidates < 2:
                relative_gpt_score = 0.0
            else:
                best_other = max(scores[i] for i in range(n_candidates) if i != resolved_chosen_idx)
                relative_gpt_score = generation_score - best_other
            break
        except Exception as exc:
            eval_retries += 1
            if eval_retries > score_cfg.max_retries:
                raise ValueError(f"Failed to get valid generation score after {score_cfg.max_retries} retries: {exc}")

    return {
        "accuracy": accuracy,
        "ranking_score": ranking_score,
        "generation_score": generation_score,
        "relative_gpt_score": relative_gpt_score,
        "predicted_idx": top_choice_idx,
        "actual_idx": resolved_chosen_idx,
    }


def aggregate_per_turn(turn_metrics: Dict[int, List[float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for turn_idx in sorted(turn_metrics.keys()):
        vals = turn_metrics[turn_idx]
        out[str(turn_idx)] = {
            "mean": mean(vals),
            "count": len(vals),
        }
    return out


def aggregate_per_turn_full_length_users(results: List[Dict]) -> Tuple[Dict[str, Dict[str, float]], int, int]:
    """
    Aggregate per-turn metrics using only users that reach the global max turn.

    Returns:
      per_turn_stats: dict keyed by turn index string
      max_turn_index: global max turn index
      n_full_length_users: number of users included
    """
    if not results:
        return {}, -1, 0

    user_max_turn = {}
    for user_res in results:
        turns = [tr["turn"] for tr in user_res.get("turn_results", [])]
        user_max_turn[user_res["user_id"]] = max(turns) if turns else -1

    max_turn_index = max(user_max_turn.values()) if user_max_turn else -1
    full_users = {uid for uid, mt in user_max_turn.items() if mt == max_turn_index}

    per_turn_acc = defaultdict(list)
    per_turn_rank = defaultdict(list)
    per_turn_gen = defaultdict(list)
    per_turn_rel = defaultdict(list)

    for user_res in results:
        if user_res["user_id"] not in full_users:
            continue
        for tr in user_res.get("turn_results", []):
            t = tr["turn"]
            per_turn_acc[t].append(tr["accuracy"])
            per_turn_rank[t].append(tr["ranking_score"])
            per_turn_gen[t].append(tr["generation_score"])
            per_turn_rel[t].append(tr["relative_gpt_score"])

    per_turn_stats = {}
    for t in range(max_turn_index + 1):
        acc_vals = per_turn_acc.get(t, [])
        rank_vals = per_turn_rank.get(t, [])
        gen_vals = per_turn_gen.get(t, [])
        rel_vals = per_turn_rel.get(t, [])
        if not acc_vals or not rank_vals or not gen_vals or not rel_vals:
            continue
        per_turn_stats[str(t)] = {
            "accuracy": mean(acc_vals),
            "ranking_score": mean(rank_vals),
            "generation_score": mean(gen_vals),
            "relative_gpt_score": mean(rel_vals),
            "count": len(acc_vals),
        }

    return per_turn_stats, max_turn_index, len(full_users)


def run_baseline(args):
    random.seed(args.seed)

    reasoning_model = build_model(
        backend=args.reasoning_backend,
        model_name=args.reasoning_model,
        base_url=args.reasoning_base_url,
        api_key=args.reasoning_api_key,
        hf_enable_thinking=args.hf_enable_thinking,
    )
    scoring_model = build_model(
        backend=args.score_backend,
        model_name=args.score_model,
        base_url=args.score_base_url,
        api_key=args.score_api_key,
        hf_enable_thinking=False,
    )

    reasoning_cfg = GenerationConfig(
        model=args.reasoning_model,
        max_tokens=args.reasoning_max_tokens,
        temperature=0.0,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        reasoning_effort=args.reasoning_effort,
    )
    score_cfg = GenerationConfig(
        model=args.score_model,
        max_tokens=args.score_max_tokens,
        temperature=0.0,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        reasoning_effort=args.score_reasoning_effort,
    )

    loaded_users = load_data(args.dataset, n_users=args.n_users, seed=args.seed)
    users = loaded_users[: args.users_per_run] if args.users_per_run is not None else loaded_users

    all_acc, all_rank, all_gen, all_rel = [], [], [], []
    skipped_unlabeled_turns = 0
    per_turn_acc = defaultdict(list)
    per_turn_rank = defaultdict(list)
    per_turn_gen = defaultdict(list)
    per_turn_rel = defaultdict(list)
    results = []

    for user in users:
        user_result = {"user_id": user.user_id, "turn_results": []}
        for conv in user.conversations:
            conversation_history: List[Turn] = []
            for ti, turn in enumerate(conv.turns):
                resolved_chosen_idx = _resolve_chosen_idx(turn)
                if resolved_chosen_idx < 0:
                    skipped_unlabeled_turns += 1
                    continue
                if resolved_chosen_idx != turn.chosen_idx:
                    turn.chosen_idx = resolved_chosen_idx

                conversation_history.append(turn)
                metrics = predict_ranking_and_metrics(
                    reasoning_model=reasoning_model,
                    scoring_model=scoring_model,
                    conversation_history=conversation_history,
                    reasoning_cfg=reasoning_cfg,
                    score_cfg=score_cfg,
                    max_history_turns=args.max_history_turns,
                    max_chars=args.max_chars,
                    ranking_fail_mode=args.ranking_fail_mode,
                )

                user_result["turn_results"].append({"turn": ti, **metrics})

                all_acc.append(metrics["accuracy"])
                all_rank.append(metrics["ranking_score"])
                all_gen.append(metrics["generation_score"])
                all_rel.append(metrics["relative_gpt_score"])

                per_turn_acc[ti].append(metrics["accuracy"])
                per_turn_rank[ti].append(metrics["ranking_score"])
                per_turn_gen[ti].append(metrics["generation_score"])
                per_turn_rel[ti].append(metrics["relative_gpt_score"])

        results.append(user_result)

    overall = {
        "accuracy": mean(all_acc) if all_acc else 0.0,
        "ranking_score": mean(all_rank) if all_rank else 0.0,
        "generation_score": mean(all_gen) if all_gen else 0.0,
        "relative_gpt_score": mean(all_rel) if all_rel else 0.0,
        "n_loaded_users": len(loaded_users),
        "n_users": len(users),
        "n_turns": len(all_acc),
        "n_skipped_unlabeled_turns": skipped_unlabeled_turns,
    }

    analysis_summary = {
        "turn_accuracy": aggregate_per_turn(per_turn_acc),
        "turn_ranking_score": aggregate_per_turn(per_turn_rank),
        "turn_generation_score": aggregate_per_turn(per_turn_gen),
        "turn_relative_gpt_score": aggregate_per_turn(per_turn_rel),
    }

    # Your requested trend view: only users with the longest turn length.
    full_turn_stats, max_turn_index, n_full_users = aggregate_per_turn_full_length_users(results)
    analysis_summary["turn_trend_full_length_users"] = {
        "max_turn_index": max_turn_index,
        "max_turn": max_turn_index + 1 if max_turn_index >= 0 else 0,
        "n_full_length_users": n_full_users,
        "per_turn": full_turn_stats,
    }

    summary = {"overall": overall, "avg_per_turn": analysis_summary}

    run_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    results_path = os.path.join(run_dir, "results.json")
    summary_path = os.path.join(run_dir, "summary.json")
    analysis_path = os.path.join(run_dir, "analysis_summary.json")
    trend_plot_path = os.path.join(run_dir, "turn_metric_trends_full_length_users.png")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)

    _plot_turn_trends(full_turn_stats, trend_plot_path)

    print("=== CoT Baseline Complete ===")
    print(json.dumps(overall, indent=2))
    print(f"Saved results: {results_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved per-turn metrics: {analysis_path}")
    print(f"Saved turn trend plot: {trend_plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run simple CoT baseline without preference tracing")
    parser.add_argument("--dataset", type=str, default="prism")
    parser.add_argument("--n-users", type=int, default=1000)
    parser.add_argument("--users-per-run", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--reasoning-backend", type=str, choices=["hf", "openai"], default="hf")
    parser.add_argument("--reasoning-model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--reasoning-base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--reasoning-api-key", type=str, default=None)
    parser.add_argument("--reasoning-max-tokens", type=int, default=256)
    parser.add_argument("--reasoning-effort", type=str, default="minimal")

    parser.add_argument("--score-backend", type=str, choices=["hf", "openai"], default="openai")
    parser.add_argument("--score-model", type=str, default="gpt-5")
    parser.add_argument("--score-base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--score-api-key", type=str, default=None)
    parser.add_argument("--score-max-tokens", type=int, default=256)
    parser.add_argument("--score-reasoning-effort", type=str, default="minimal")

    parser.add_argument("--max-history-turns", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=280)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=0.5)
    parser.add_argument("--ranking-fail-mode", type=str, choices=["raise", "fallback"], default="fallback")

    parser.add_argument("--hf-enable-thinking", action="store_true")

    parser.add_argument("--output-dir", type=str, default="baseline_results")
    parser.add_argument("--run-id", type=str, default="cot_qwen35_4b")
    return parser.parse_args()


if __name__ == "__main__":
    run_baseline(parse_args())
