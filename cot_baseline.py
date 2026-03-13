import argparse
import json
import random
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Tuple

from data import Turn, load_data
from model import GenerationConfig, HFModel, OpenAIModel


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
You are evaluating whether an Adapted response aligns more closely with the user's chosen response.

Given:
- Current user message
- Candidate responses (with chosen or rejected specified)
- Adapted response (to evaluate)

Goal:
Determine whether the Adapted response is more similar to the Chosen candidate than to the Rejected candidate.

Step 1:
Identify key preference-relevant differences between Chosen and Rejected.
Focus on (not exhaustive, and only include if relevant):
- Underlying value
- Tone
- Structure
- Information density
- Level of abstraction
- Actionability
- Framing

Step 2:
Compare the Adapted response to both candidates along those dimensions.

Evaluation principles:
- Only consider dimensions that distinguish Chosen from Rejected.
- Do not assume hidden user traits.

Scoring rubric (1-10):

9-10: Adapted clearly reflects the distinguishing qualities of the Chosen candidate.
7-8: Adapted mostly reflects Chosen, with minor resemblance to Rejected.
5-6: Mixed; partially resembles both.
3-4: Adapted resembles Rejected more on key distinguishing aspects.
1-2: Adapted strongly resembles Rejected.

Output JSON only:
{
  "reason": "2-3 sentences describing the key distinguishing signals and how Adapted compares.",
  "score": 1-10
}

[Interaction]
{current_turn}

[Adapted response]
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
    left = text.find("{")
    right = text.rfind("}")
    if left == -1 or right == -1 or right < left:
        raise ValueError(f"No JSON object found in output: {text}")
    return json.loads(text[left : right + 1])


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
) -> Dict[str, float]:
    current_turn = conversation_history[-1]
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
    while True:
        try:
            output = reasoning_model.generate(prompt, cfg=reasoning_cfg)["output"]
            ranking_json = _extract_json(output)
            ranking = _normalize_ranking(ranking_json.get("ranking"), n_candidates=len(current_turn.candidates))
            break
        except Exception as exc:
            retries += 1
            if retries > reasoning_cfg.max_retries:
                raise ValueError(f"Failed to get valid ranking after {reasoning_cfg.max_retries} retries: {exc}")

    gt_choice = current_turn.chosen_idx + 1
    rank = ranking.index(gt_choice) + 1
    ranking_score = (len(ranking) - rank) / (len(ranking) - 1)
    accuracy = 1.0 if rank == 1 else 0.0

    top_choice_idx = ranking[0] - 1
    adapted_response = current_turn.candidates[top_choice_idx]
    eval_prompt = EVALUATE_PROMPT.format(
        current_turn=current_turn.format(include_candidates=True, include_choice=True),
        adapted=adapted_response,
    )

    eval_retries = 0
    while True:
        try:
            score_output = scoring_model.generate(eval_prompt, cfg=score_cfg)["output"]
            score_json = _extract_json(score_output)
            generation_score = float(score_json["score"])
            break
        except Exception as exc:
            eval_retries += 1
            if eval_retries > score_cfg.max_retries:
                raise ValueError(f"Failed to get valid generation score after {score_cfg.max_retries} retries: {exc}")

    return {
        "accuracy": accuracy,
        "ranking_score": ranking_score,
        "generation_score": generation_score,
        "predicted_idx": top_choice_idx,
        "actual_idx": current_turn.chosen_idx,
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

    all_acc, all_rank, all_gen = [], [], []
    per_turn_acc = defaultdict(list)
    per_turn_rank = defaultdict(list)
    per_turn_gen = defaultdict(list)
    results = []

    for user in users:
        user_result = {"user_id": user.user_id, "turn_results": []}
        for conv in user.conversations:
            conversation_history: List[Turn] = []
            for ti, turn in enumerate(conv.turns):
                conversation_history.append(turn)
                metrics = predict_ranking_and_metrics(
                    reasoning_model=reasoning_model,
                    scoring_model=scoring_model,
                    conversation_history=conversation_history,
                    reasoning_cfg=reasoning_cfg,
                    score_cfg=score_cfg,
                    max_history_turns=args.max_history_turns,
                    max_chars=args.max_chars,
                )

                user_result["turn_results"].append({"turn": ti, **metrics})

                all_acc.append(metrics["accuracy"])
                all_rank.append(metrics["ranking_score"])
                all_gen.append(metrics["generation_score"])

                per_turn_acc[ti].append(metrics["accuracy"])
                per_turn_rank[ti].append(metrics["ranking_score"])
                per_turn_gen[ti].append(metrics["generation_score"])

        results.append(user_result)

    overall = {
        "accuracy": mean(all_acc) if all_acc else 0.0,
        "ranking_score": mean(all_rank) if all_rank else 0.0,
        "generation_score": mean(all_gen) if all_gen else 0.0,
        "n_loaded_users": len(loaded_users),
        "n_users": len(users),
        "n_turns": len(all_acc),
    }

    analysis_summary = {
        "turn_accuracy": aggregate_per_turn(per_turn_acc),
        "turn_ranking_score": aggregate_per_turn(per_turn_rank),
        "turn_generation_score": aggregate_per_turn(per_turn_gen),
    }

    summary = {"overall": overall, "avg_per_turn": analysis_summary}

    import os

    run_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    results_path = os.path.join(run_dir, "results.json")
    summary_path = os.path.join(run_dir, "summary.json")
    analysis_path = os.path.join(run_dir, "analysis_summary.json")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)

    print("=== CoT Baseline Complete ===")
    print(json.dumps(overall, indent=2))
    print(f"Saved results: {results_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved per-turn metrics: {analysis_path}")


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

    parser.add_argument("--hf-enable-thinking", action="store_true")

    parser.add_argument("--output-dir", type=str, default="baseline_results")
    parser.add_argument("--run-id", type=str, default="cot_qwen35_4b")
    return parser.parse_args()


if __name__ == "__main__":
    run_baseline(parse_args())
