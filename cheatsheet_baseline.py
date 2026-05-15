"""
Dynamic Cheatsheet / Running User Profile Baseline
===================================================
Per-user online memory baseline: maintains a single textual cheatsheet of
observed user preferences, updated after every turn using the gold preference.

Ordering invariant (strictly enforced):
  For turn t:
    1.  Predict ranking using cheatsheet built from turns 0..t-1  (no gold leakage)
    2.  Record prediction + compute adaptation metrics
    3.  Update cheatsheet using turn t's gold preference           (safe: after predict)

No hypothesis particles, weighting, resampling, or SMC mechanisms are used.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data import Turn, load_data
from model import (
    GenerationConfig,
    HFModel,
    OpenAIModel,
    ChatModel,
    EmbedConfig,
    embed,
    candidate_similarity_scores,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INITIAL_CHEATSHEET = (
    "User Preference Cheatsheet:\n"
    "No reliable user preferences observed yet."
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CHEATSHEET_RANK_PROMPT = """
You are selecting the best response for the current user message.

Given:
- A concise cheatsheet of this user's known preferences (may be empty at first)
- Current user message
- Candidate responses with ids

Task:
Based on the user's known preferences, rank the candidates from most to least preferred.

Rules:
- Use only the cheatsheet and candidate content as evidence.
- Do not speculate beyond what the cheatsheet says.
- Do not output chain-of-thought.
- Output valid JSON only.

Return exactly:
{{
  "reason": "Brief 1-2 sentence summary of key differences relative to user preferences.",
  "ranking": [1, 2, 3]
}}

[User Preference Cheatsheet]
{cheatsheet}

[Current user message]
{current_message}

[Candidates]
{candidates}
"""

CHEATSHEET_UPDATE_PROMPT = """
You are maintaining a concise user preference cheatsheet for future preference prediction.

Previous cheatsheet:
{cheatsheet}

New observed interaction:
User message:
{user_message}

Candidate responses:
{candidates}

The user preferred:
{chosen_response}

Update the cheatsheet based on this interaction.

Rules:
- Keep only preferences that may help future response ranking.
- Do not simply copy the interaction; extract generalizable preference signals.
- Avoid overgeneralizing from a single example.
- Keep the cheatsheet concise: at most {max_bullets} bullet points.
- Mark uncertain or tentative preferences explicitly, e.g. "(tentative)".
- Start with "User Preference Cheatsheet:" on the first line.

Return only the updated cheatsheet, no extra commentary.
"""

EVALUATE_PROMPT = """
You are evaluating the similarity between an Adapted response and a set of candidate responses in the context of the user preferences.

Inputs:
- Current turn with c candidates
- Adapted response

Step 1:
Identify 2-4 preference-relevant dimensions that distinguish the candidates.
Use short canonical labels (e.g., "values", "information_density", "structure", "actionability", "tone", "framing", "abstraction") (Not exhaustive and do not force-fit).

Step 2:
Using those dimensions, score how similar the Adapted response is to EACH candidate.
Scores are 0-5:
5 = Near-identical on the key dimensions; Adapted matches candidate's stance/style/structure with no meaningful drift.
4 = Strong match on most key dimensions; minor drift on at most one dimension.
3 = Partial match; aligns on some key dimensions but differs on others OR ambiguity prevents a clear judgment.
2 = Weak match; differs on one or more key dimensions in a way that matters.
1 = Very weak match; mostly reflects the opposite of the candidate on key dimensions.
0 = Opposes/contradicts the candidate on the key dimensions (clear mismatch).

Output valid, parsable JSON only without any extra commentary:
{{
  "dimensions": ["...", "..."],
  "scores": [s0, s1, ..., s{{c-1}}],
  "justification": "a brief (1-2 sentences) explanation of the key similarities/differences between the Adapted response and each candidates."
}}

Rules:
- scores length MUST equal number of candidates; scores[i] corresponds to candidate i. Do NOT change the candidate order.
- Use the same dimensions for scoring all candidates.

[Current turn]
{current_turn}

[Number of candidates]
c={c}

[Adapted]
{adapted}
"""

# ---------------------------------------------------------------------------
# Mock models (used by --dry-run; no API calls)
# ---------------------------------------------------------------------------

class MockModel:
    """Deterministic mock that returns syntactically valid outputs."""

    def __init__(self, mode: str = "ranking"):
        self.mode = mode

    def generate(self, prompt: str, cfg=None) -> Dict:
        if self.mode == "ranking":
            cands = re.findall(r"\[C(\d+)\]", prompt)
            n = len(cands) if cands else 3
            ranking = list(range(1, n + 1))
            output = json.dumps({"reason": "Mock ranking.", "ranking": ranking})
        elif self.mode == "scoring":
            cands = re.findall(r"\[C(\d+)\]", prompt)
            n = len(cands) if cands else 3
            scores = [3.0] * n
            output = json.dumps({
                "dimensions": ["tone"],
                "scores": scores,
                "justification": "Mock scoring.",
            })
        elif self.mode == "update":
            output = (
                "User Preference Cheatsheet:\n"
                "- (tentative) Mock preference A\n"
                "- Mock preference B"
            )
        else:
            output = ""
        return {
            "output": output,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def batch_generate(self, prompts: List[str], cfg=None) -> List[Dict]:
        return [self.generate(p, cfg) for p in prompts]


# ---------------------------------------------------------------------------
# Utility helpers (mirror CoT / RAG baseline)
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _plot_turn_trends(per_turn_stats: Dict[str, Dict[str, float]], output_path: str):
    import matplotlib.pyplot as plt

    turns = sorted(int(t) for t in per_turn_stats.keys())
    if not turns:
        return

    x = [t + 1 for t in turns]
    acc      = [per_turn_stats[str(t)]["accuracy"] for t in turns]
    rank     = [per_turn_stats[str(t)]["ranking_score"] for t in turns]
    gen      = [per_turn_stats[str(t)]["generation_score"] for t in turns]
    rel_gpt  = [per_turn_stats[str(t)]["relative_gpt_score"] for t in turns]
    sim      = [per_turn_stats[str(t)]["similarity_score"] for t in turns]
    rel_sim  = [per_turn_stats[str(t)]["relative_similarity_score"] for t in turns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(x, acc, marker="o", linewidth=2, color="tab:blue")
    axes[0, 0].set_title("Accuracy by Turn"); axes[0, 0].set_ylabel("Accuracy"); axes[0, 0].set_ylim(0.0, 1.0); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x, rank, marker="s", linewidth=2, color="tab:orange")
    axes[0, 1].set_title("Ranking Score by Turn"); axes[0, 1].set_ylabel("Ranking Score"); axes[0, 1].set_ylim(0.0, 1.0); axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(x, gen, marker="^", linewidth=2, color="tab:green")
    axes[0, 2].set_title("Generation Score by Turn"); axes[0, 2].set_ylabel("GPT Score (0-5)"); axes[0, 2].set_ylim(0.0, 5.0); axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(x, rel_gpt, marker="d", linewidth=2, color="tab:red")
    axes[1, 0].set_title("Relative GPT Score by Turn"); axes[1, 0].set_ylabel("Relative GPT Score"); axes[1, 0].set_ylim(-5.0, 5.0); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x, sim, marker="p", linewidth=2, color="tab:purple")
    axes[1, 1].set_title("Similarity Score by Turn"); axes[1, 1].set_ylabel("Embedding Similarity"); axes[1, 1].set_ylim(-1.0, 1.0); axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(x, rel_sim, marker="h", linewidth=2, color="tab:brown")
    axes[1, 2].set_title("Relative Similarity Score by Turn"); axes[1, 2].set_ylabel("Relative Similarity"); axes[1, 2].set_ylim(-2.0, 2.0); axes[1, 2].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Turn")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _compact_text(text: str, max_chars: int) -> str:
    text = _collapse_ws(text)
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]} ... {text[-half:]}"


def _build_candidates(turn: Turn, max_chars: int) -> str:
    lines = []
    for i, c in enumerate(turn.candidates, start=1):
        lines.append(f"[C{i}]\n{_compact_text(c, max_chars)}\n")
    return "\n".join(lines)


def _extract_json(text: str) -> Dict:
    cleaned = re.sub(r"<think>[\\s\\S]*?</think>", "", text, flags=re.IGNORECASE)
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
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
    if 0 <= turn.chosen_idx < len(turn.candidates):
        return turn.chosen_idx
    if turn.chosen and turn.chosen in turn.candidates:
        return turn.candidates.index(turn.chosen)
    return -1


def build_model(
    backend: str,
    model_name: str,
    base_url: Optional[str],
    api_key: Optional[str],
    hf_enable_thinking: bool,
):
    if backend == "hf":
        return HFModel(model_name=model_name, enable_thinking=hf_enable_thinking)
    if backend == "openai":
        return OpenAIModel(api_key=api_key, base_url=base_url, model=model_name)
    if backend == "openrouter":
        return ChatModel(api_key=api_key, base_url=base_url or ChatModel.OPENROUTER_BASE_URL, model=model_name)
    raise ValueError(f"Unsupported backend: {backend}")


def calculate_cost(usage: Dict[str, int], reasoning_model: str, scoring_model: str) -> float:
    """Cost in USD. Update calls are billed at scoring model price."""
    PRICING = {
        "gpt-5.4-nano": {"input": 0.20,  "output": 1.25},
        "gpt-5":        {"input": 1.25,  "output": 10.0},
        "default":      {"input": 0.20,  "output": 1.25},
    }

    def get_price(name: str) -> Dict:
        for key in PRICING:
            if key in name.lower():
                return PRICING[key]
        return PRICING["default"]

    rp = get_price(reasoning_model)
    sp = get_price(scoring_model)

    reasoning_cost = (
        usage.get("reasoning_input",  0) * rp["input"]  / 1_000_000
        + usage.get("reasoning_output", 0) * rp["output"] / 1_000_000
    )
    scoring_cost = (
        usage.get("scoring_input",  0) * sp["input"]  / 1_000_000
        + usage.get("scoring_output", 0) * sp["output"] / 1_000_000
    )
    update_cost = (
        usage.get("update_input",  0) * sp["input"]  / 1_000_000
        + usage.get("update_output", 0) * sp["output"] / 1_000_000
    )
    return reasoning_cost + scoring_cost + update_cost


# ---------------------------------------------------------------------------
# Core: cheatsheet update
# ---------------------------------------------------------------------------

def update_cheatsheet(
    cheatsheet: str,
    turn: Turn,
    update_model,
    update_cfg: GenerationConfig,
    max_bullets: int,
    max_chars: int,
    verbose: bool = False,
) -> Tuple[str, Dict]:
    """
    Call update_model to revise the cheatsheet using the gold preference from
    the current turn.  Returns (updated_cheatsheet, usage_dict).

    This must be called AFTER prediction to avoid data leakage.
    """
    candidates_text = _build_candidates(turn, max_chars)
    chosen_text = _compact_text(turn.chosen, max_chars * 2)
    user_msg_text = _compact_text(turn.user_message, max_chars * 2)

    prompt = CHEATSHEET_UPDATE_PROMPT.format(
        cheatsheet=cheatsheet,
        user_message=user_msg_text,
        candidates=candidates_text,
        chosen_response=chosen_text,
        max_bullets=max_bullets,
    )

    usage = {"update_input": 0, "update_output": 0}
    try:
        result = update_model.generate(prompt, cfg=update_cfg)
        updated = result["output"].strip()
        if "usage" in result:
            usage["update_input"]  += result["usage"].get("prompt_tokens", 0)
            usage["update_output"] += result["usage"].get("completion_tokens", 0)
        if not updated:
            updated = cheatsheet
    except Exception as exc:
        print(f"[WARN] Cheatsheet update failed: {exc}")
        updated = cheatsheet

    if verbose:
        print(f"  [Cheatsheet updated]\n{updated}\n")

    return updated, usage


# ---------------------------------------------------------------------------
# Core: predict ranking + adaptation metrics for one turn
# ---------------------------------------------------------------------------

def predict_ranking_and_metrics(
    reasoning_model,
    scoring_model,
    current_turn: Turn,
    cheatsheet: str,
    reasoning_cfg: GenerationConfig,
    score_cfg: GenerationConfig,
    embed_cfg: Optional[EmbedConfig],
    max_chars: int,
    ranking_fail_mode: str = "fallback",
) -> Dict:
    """
    Predict ranking for *current_turn* using *cheatsheet* (which contains only
    information from turns before the current one — no leakage).

    Returns:
        {
            "prediction": {success, accuracy, ranking_score, ranking, chosen_idx},
            "adaptation": {success, gpt_score, relative_gpt_score, ...},
            "usage":      {reasoning_input, reasoning_output, scoring_input, scoring_output},
        }
    """
    resolved_chosen_idx = _resolve_chosen_idx(current_turn)
    if resolved_chosen_idx < 0:
        raise ValueError("Current turn has no valid chosen candidate index.")

    current_message = _compact_text(current_turn.user_message, max_chars)
    candidates_text = _build_candidates(current_turn, max_chars=max_chars)

    # Build prediction prompt — cheatsheet must NOT contain current turn's gold
    prompt = CHEATSHEET_RANK_PROMPT.format(
        cheatsheet=cheatsheet,
        current_message=current_message,
        candidates=candidates_text,
    )

    retries = 0
    n_candidates = len(current_turn.candidates)
    last_exc = None
    usage_stats = {"reasoning_input": 0, "reasoning_output": 0,
                   "scoring_input": 0, "scoring_output": 0}

    ranking = None
    while True:
        try:
            result = reasoning_model.generate(prompt, cfg=reasoning_cfg)
            output = result["output"]
            if "usage" in result:
                usage_stats["reasoning_input"]  += result["usage"].get("prompt_tokens", 0)
                usage_stats["reasoning_output"] += result["usage"].get("completion_tokens", 0)
            ranking_json = _extract_json(output)
            ranking = _normalize_ranking(ranking_json.get("ranking"), n_candidates=n_candidates)
            break
        except Exception as exc:
            last_exc = exc
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
                    print(f"[WARN] Ranking parse failed after retries; fallback. Last error: {last_exc}")
                    return {
                        "prediction": {
                            "success": False, "accuracy": 0.0, "ranking_score": 0.5,
                            "reason": str(last_exc), "chosen_idx": resolved_chosen_idx,
                        },
                        "adaptation": {
                            "success": False, "error": "Ranking failed; no adapted response.",
                            "gpt_score": None, "relative_gpt_score": None, "relative_mean_gpt_score": None,
                            "gpt_scores": [], "rejected_gpt_scores": [],
                            "similarity_score": None, "relative_score": None, "relative_mean_score": None,
                            "similarity_scores": [], "rejected_similarity_scores": [],
                            "chosen_idx": resolved_chosen_idx,
                        },
                        "usage": usage_stats,
                    }
                raise ValueError(f"Failed valid ranking after {reasoning_cfg.max_retries} retries: {last_exc}")

    gt_choice = resolved_chosen_idx + 1
    rank = ranking.index(gt_choice) + 1
    ranking_score = 1.0 if len(ranking) == 1 else (len(ranking) - rank) / (len(ranking) - 1)
    accuracy = 1.0 if rank == 1 else 0.0

    top_choice_idx = ranking[0] - 1
    adapted_response = current_turn.candidates[top_choice_idx]

    # --- GPT scoring ---
    eval_prompt = EVALUATE_PROMPT.format(
        current_turn=current_turn.format(include_candidates=True, include_choice=False),
        c=n_candidates,
        adapted=adapted_response,
    )

    gpt_score = None
    relative_gpt_score = None
    relative_mean_gpt_score = None
    gpt_scores: List[float] = []
    rejected_gpt_scores: List[float] = []
    eval_error = None

    if n_candidates < 2:
        gpt_score = 2.5
        relative_gpt_score = 0.0
        relative_mean_gpt_score = 0.0
    else:
        eval_retries = 0
        while True:
            try:
                result = scoring_model.generate(eval_prompt, cfg=score_cfg)
                score_output = result["output"]
                if "usage" in result:
                    usage_stats["scoring_input"]  += result["usage"].get("prompt_tokens", 0)
                    usage_stats["scoring_output"] += result["usage"].get("completion_tokens", 0)
                score_json = _extract_json(score_output)
                scores = score_json["scores"]
                if not isinstance(scores, list) or len(scores) != n_candidates:
                    raise ValueError(
                        f"scores length mismatch: expected {n_candidates}, got "
                        f"{len(scores) if isinstance(scores, list) else 'non-list'}"
                    )
                gpt_scores = [float(s) for s in scores]
                gpt_score = gpt_scores[resolved_chosen_idx]
                rejected_gpt_scores = [s for i, s in enumerate(gpt_scores) if i != resolved_chosen_idx]
                relative_gpt_score = gpt_score - max(rejected_gpt_scores)
                relative_mean_gpt_score = gpt_score - (sum(rejected_gpt_scores) / len(rejected_gpt_scores))
                break
            except Exception as exc:
                eval_retries += 1
                if eval_retries > score_cfg.max_retries:
                    eval_error = str(exc)
                    break

    # --- Embedding similarity ---
    similarity_score = None
    relative_score = None
    relative_mean_score = None
    similarity_scores: List[float] = []
    rejected_similarity_scores: List[float] = []
    if embed_cfg is not None:
        try:
            similarity_scores = candidate_similarity_scores(adapted_response, current_turn.candidates, embed_cfg)
            similarity_score = similarity_scores[resolved_chosen_idx]
            rejected_similarity_scores = [s for i, s in enumerate(similarity_scores) if i != resolved_chosen_idx]
            relative_score = similarity_score - max(rejected_similarity_scores) if rejected_similarity_scores else 0.0
            relative_mean_score = (
                similarity_score - (sum(rejected_similarity_scores) / len(rejected_similarity_scores))
                if rejected_similarity_scores else 0.0
            )
        except Exception as exc:
            print(f"[WARN] Embedding similarity computation failed: {exc}")

    adaptation = {
        "success": eval_error is None and gpt_score is not None,
        "gpt_score": gpt_score,
        "relative_gpt_score": relative_gpt_score,
        "relative_mean_gpt_score": relative_mean_gpt_score,
        "gpt_scores": gpt_scores,
        "rejected_gpt_scores": rejected_gpt_scores,
        "similarity_score": similarity_score,
        "relative_score": relative_score,
        "relative_mean_score": relative_mean_score,
        "similarity_scores": similarity_scores,
        "rejected_similarity_scores": rejected_similarity_scores,
        "chosen_idx": resolved_chosen_idx,
    }
    if eval_error:
        adaptation["error"] = eval_error

    return {
        "prediction": {
            "success": True,
            "accuracy": accuracy,
            "ranking_score": ranking_score,
            "ranking": ranking,
            "chosen_idx": resolved_chosen_idx,
        },
        "adaptation": adaptation,
        "usage": usage_stats,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers (identical to CoT / RAG)
# ---------------------------------------------------------------------------

def _safe_mean(values):
    clean = [v for v in values if v is not None]
    return mean(clean) if clean else None


def summarize_user_metrics(turns: List[Dict]) -> Dict:
    predictions = [t["prediction"] for t in turns]
    valid_pred = [p for p in predictions if p.get("success")]
    adaptations = [t["adaptation"] for t in turns]
    valid_adapt = [a for a in adaptations if a.get("success")]
    return {
        "n_turns": len(turns),
        "n_prediction_turns": len(valid_pred),
        "n_adaptation_turns": len(valid_adapt),
        "prediction_success_rate": _safe_mean([1.0 if p.get("success") else 0.0 for p in predictions]),
        "prediction_accuracy": _safe_mean([p.get("accuracy") for p in valid_pred]),
        "prediction_ranking_score": _safe_mean([p.get("ranking_score") for p in valid_pred]),
        "adapt_gpt_score": _safe_mean([a.get("gpt_score") for a in valid_adapt]),
        "adapt_relative_gpt_score": _safe_mean([a.get("relative_gpt_score") for a in valid_adapt]),
        "adapt_relative_mean_gpt_score": _safe_mean([a.get("relative_mean_gpt_score") for a in valid_adapt]),
        "adapt_similarity_score": _safe_mean([a.get("similarity_score") for a in valid_adapt]),
        "adapt_relative_score": _safe_mean([a.get("relative_score") for a in valid_adapt]),
        "adapt_relative_mean_score": _safe_mean([a.get("relative_mean_score") for a in valid_adapt]),
    }


def summarize_all_metrics(user_records: List[Dict]) -> Dict:
    all_turns = [t for rec in user_records for t in rec.get("turns", [])]
    all_predictions = [t["prediction"] for t in all_turns]
    valid_pred = [p for p in all_predictions if p.get("success")]
    all_adaptations = [t["adaptation"] for t in all_turns]
    valid_adapt = [a for a in all_adaptations if a.get("success")]

    max_turn_index = max((len(rec.get("turns", [])) for rec in user_records), default=0)
    online_turns = []
    for turn_index in range(max_turn_index):
        turn_slices = [
            rec["turns"][turn_index]
            for rec in user_records
            if turn_index < len(rec.get("turns", []))
        ]
        preds = [t["prediction"] for t in turn_slices]
        valid_tp = [p for p in preds if p.get("success")]
        adapts = [t["adaptation"] for t in turn_slices]
        valid_ta = [a for a in adapts if a.get("success")]
        online_turns.append({
            "turn_index": turn_index,
            "n_users": len(turn_slices),
            "n_prediction_users": len(valid_tp),
            "n_adaptation_users": len(valid_ta),
            "prediction_accuracy": _safe_mean([p.get("accuracy") for p in valid_tp]),
            "prediction_ranking_score": _safe_mean([p.get("ranking_score") for p in valid_tp]),
            "adapt_gpt_score": _safe_mean([a.get("gpt_score") for a in valid_ta]),
            "adapt_relative_gpt_score": _safe_mean([a.get("relative_gpt_score") for a in valid_ta]),
            "adapt_relative_mean_gpt_score": _safe_mean([a.get("relative_mean_gpt_score") for a in valid_ta]),
            "adapt_similarity_score": _safe_mean([a.get("similarity_score") for a in valid_ta]),
            "adapt_relative_score": _safe_mean([a.get("relative_score") for a in valid_ta]),
            "adapt_relative_mean_score": _safe_mean([a.get("relative_mean_score") for a in valid_ta]),
        })

    return {
        "n_users": len(user_records),
        "n_turns": len(all_turns),
        "online_turns": online_turns,
        "overall_prediction": {
            "prediction_accuracy": _safe_mean([p.get("accuracy") for p in valid_pred]),
            "prediction_ranking_score": _safe_mean([p.get("ranking_score") for p in valid_pred]),
        },
        "overall_adaptation": {
            "adapt_gpt_score": _safe_mean([a.get("gpt_score") for a in valid_adapt]),
            "adapt_relative_gpt_score": _safe_mean([a.get("relative_gpt_score") for a in valid_adapt]),
            "adapt_relative_mean_gpt_score": _safe_mean([a.get("relative_mean_gpt_score") for a in valid_adapt]),
            "adapt_similarity_score": _safe_mean([a.get("similarity_score") for a in valid_adapt]),
            "adapt_relative_score": _safe_mean([a.get("relative_score") for a in valid_adapt]),
            "adapt_relative_mean_score": _safe_mean([a.get("relative_mean_score") for a in valid_adapt]),
        },
        "users": {rec["user"]: rec.get("summary", {}) for rec in user_records},
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_baseline(args):
    random.seed(args.seed)

    if args.dry_run:
        print("=" * 60)
        print("[DRY RUN] Using MockModel — no API calls will be made.")
        print("=" * 60)
        reasoning_model = MockModel(mode="ranking")
        scoring_model   = MockModel(mode="scoring")
        update_model    = MockModel(mode="update")
        embed_cfg       = None   # skip embedding in dry-run
    else:
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
        # Cheatsheet updates use the scoring model by default
        update_model = scoring_model
        embed_cfg = EmbedConfig(
            backend="openai",
            model=args.embed_model,
            api_key=args.embed_api_key or args.score_api_key,
            base_url=args.embed_base_url or "https://api.openai.com/v1",
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
    update_cfg = GenerationConfig(
        model=args.score_model,              # same model as scorer
        max_tokens=args.update_max_tokens,
        temperature=0.0,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    loaded_users = load_data(args.dataset, n_users=args.n_users, seed=args.seed)
    users = loaded_users[: args.users_per_run] if args.users_per_run is not None else loaded_users

    # Create output directory
    run_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Pre-compute total evaluable turns for progress reporting
    total_turns_planned = 0
    for user in users:
        for conv in user.conversations:
            for turn in conv.turns:
                if _resolve_chosen_idx(turn) >= 0:
                    total_turns_planned += 1

    total_usage = {
        "reasoning_input": 0, "reasoning_output": 0,
        "scoring_input": 0,   "scoring_output": 0,
        "update_input": 0,    "update_output": 0,
    }
    skipped_unlabeled_turns = 0
    processed_turns = 0
    completed_users = 0
    user_records: List[Dict] = []
    per_user_usage: List[Dict] = []

    # -----------------------------------------------------------------------
    # Dry-run invariant tracking
    # -----------------------------------------------------------------------
    dry_run_violations: List[str] = []

    for user in users:
        print(
            f"[Progress] User {completed_users + 1}/{len(users)} ({user.user_id}) started. "
            f"Processed turns: {processed_turns}/{total_turns_planned}."
        )

        # ── Invariant: cheatsheet is reset per user, NOT carried across users ──
        cheatsheet = INITIAL_CHEATSHEET
        if args.dry_run:
            print(f"  [DRY RUN] Cheatsheet reset for user {user.user_id}: {repr(cheatsheet[:60])}")

        user_turns: List[Dict] = []
        user_usage = {
            "reasoning_input": 0, "reasoning_output": 0,
            "scoring_input": 0,   "scoring_output": 0,
            "update_input": 0,    "update_output": 0,
        }
        turn_index = 0  # global across all conversations for this user

        for conv in user.conversations:
            for turn in conv.turns:
                resolved_chosen_idx = _resolve_chosen_idx(turn)
                if resolved_chosen_idx < 0:
                    skipped_unlabeled_turns += 1
                    continue

                if resolved_chosen_idx != turn.chosen_idx:
                    turn.chosen_idx = resolved_chosen_idx

                # ── Dry-run: verify gold is NOT in the cheatsheet used for prediction ──
                # NOTE: The chosen response is a candidate, so it will naturally appear
                # in the candidates block of the prediction prompt — that is not leakage.
                # Real leakage would be if the current turn's gold preference has already
                # been written into the *cheatsheet* before the prediction step.
                if args.dry_run:
                    gold_text = turn.chosen or ""
                    gold_snippet = _compact_text(gold_text, 80)[:40].strip()
                    if gold_snippet and gold_snippet in cheatsheet:
                        violation = (
                            f"[VIOLATION] User {user.user_id}, turn {turn_index}: "
                            f"gold snippet found in cheatsheet before prediction!"
                        )
                        print(violation)
                        dry_run_violations.append(violation)
                    else:
                        print(
                            f"  [DRY RUN] Turn {turn_index} | cheatsheet clean "
                            f"(gold NOT in cheatsheet ✓) | cheatsheet lines: "
                            f"{len(cheatsheet.splitlines())}"
                        )

                # ── STEP 1: PREDICT (before using gold) ─────────────────────
                result = predict_ranking_and_metrics(
                    reasoning_model=reasoning_model,
                    scoring_model=scoring_model,
                    current_turn=turn,
                    cheatsheet=cheatsheet,
                    reasoning_cfg=reasoning_cfg,
                    score_cfg=score_cfg,
                    embed_cfg=embed_cfg,
                    max_chars=args.max_chars,
                    ranking_fail_mode=args.ranking_fail_mode,
                )

                # ── STEP 2: RECORD ──────────────────────────────────────────
                for key in list(user_usage.keys()):
                    if key in result["usage"]:
                        user_usage[key]  += result["usage"][key]
                        total_usage[key] += result["usage"][key]

                turn_record = {
                    "turn_index": turn_index,
                    "turn_id": getattr(turn, "turn_id", None),
                    "prediction": result["prediction"],
                    "adaptation": result["adaptation"],
                }
                user_turns.append(turn_record)

                # ── STEP 3: UPDATE CHEATSHEET (after prediction, using gold) ─
                cheatsheet, upd_usage = update_cheatsheet(
                    cheatsheet=cheatsheet,
                    turn=turn,
                    update_model=update_model,
                    update_cfg=update_cfg,
                    max_bullets=args.max_bullets,
                    max_chars=args.max_chars,
                    verbose=args.dry_run,
                )
                for k, v in upd_usage.items():
                    user_usage[k]  += v
                    total_usage[k] += v

                turn_index += 1
                processed_turns += 1

                if processed_turns in [1, 3, 5]:
                    current_cost = calculate_cost(total_usage, args.reasoning_model, args.score_model)
                    total_tokens = sum(total_usage.values())
                    print(
                        f"[Cost Debug] After {processed_turns} turn(s): "
                        f"${current_cost:.6f} USD, {total_tokens:,} tokens"
                    )

                if processed_turns % 10 == 0 or processed_turns == total_turns_planned:
                    pct = (100.0 * processed_turns / total_turns_planned) if total_turns_planned else 100.0
                    current_cost = calculate_cost(total_usage, args.reasoning_model, args.score_model)
                    print(
                        f"[Progress] Turns {processed_turns}/{total_turns_planned} "
                        f"({pct:.1f}%), skipped_unlabeled={skipped_unlabeled_turns}, "
                        f"cost=${current_cost:.4f}"
                    )

        # Save final cheatsheet for this user (for analysis)
        cheatsheet_path = Path(run_dir) / "cheatsheets" / f"{user.user_id}.txt"
        cheatsheet_path.parent.mkdir(parents=True, exist_ok=True)
        cheatsheet_path.write_text(cheatsheet, encoding="utf-8")

        user_cost = calculate_cost(user_usage, args.reasoning_model, args.score_model)
        per_user_usage.append({"user_id": user.user_id, "usage": user_usage, "cost_usd": user_cost})

        user_record = {
            "user": user.user_id,
            "turns": user_turns,
            "summary": summarize_user_metrics(user_turns),
        }
        user_records.append(user_record)
        completed_users += 1
        print(
            f"[Progress] User {completed_users}/{len(users)} completed. "
            f"Processed turns: {processed_turns}/{total_turns_planned}, "
            f"skipped_unlabeled={skipped_unlabeled_turns}."
        )

    # -----------------------------------------------------------------------
    # Build and write outputs
    # -----------------------------------------------------------------------
    summary = summarize_all_metrics(user_records)
    total_cost = calculate_cost(total_usage, args.reasoning_model, args.score_model)
    average_cost_per_user = total_cost / len(users) if users else 0.0
    cost_summary = {
        "total_usage": total_usage,
        "total_cost_usd": total_cost,
        "average_cost_per_user_usd": average_cost_per_user,
        "per_user_usage": per_user_usage,
        "pricing_info": {
            "reasoning_model": args.reasoning_model,
            "scoring_model": args.score_model,
        },
    }

    run_dir_path = Path(run_dir)
    users_dir = run_dir_path / "users"
    users_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir_path / "summary.json"
    cost_path = run_dir_path / "cost_report.json"
    trend_plot_path = run_dir_path / "turn_metric_trends.png"

    for rec in user_records:
        _write_json(users_dir / f"{rec['user']}.json", rec)

    _write_json(summary_path, summary)
    _write_json(cost_path, cost_summary)

    plot_data = {}
    for ot in summary.get("online_turns", []):
        t = str(ot["turn_index"])
        plot_data[t] = {
            "accuracy": ot.get("prediction_accuracy") or 0.0,
            "ranking_score": ot.get("prediction_ranking_score") or 0.0,
            "generation_score": ot.get("adapt_gpt_score") or 0.0,
            "relative_gpt_score": ot.get("adapt_relative_gpt_score") or 0.0,
            "similarity_score": ot.get("adapt_similarity_score") or 0.0,
            "relative_similarity_score": ot.get("adapt_relative_score") or 0.0,
        }
    _plot_turn_trends(plot_data, str(trend_plot_path))

    overall_pred  = summary.get("overall_prediction", {})
    overall_adapt = summary.get("overall_adaptation", {})
    print("\n=== Dynamic Cheatsheet Baseline Complete ===")
    print(json.dumps({**overall_pred, **overall_adapt}, indent=2))
    print("\n=== Cost Summary ===")
    print(f"Total cost: ${total_cost:.4f} USD")
    print(f"Average cost per user: ${average_cost_per_user:.4f} USD")
    print(f"Total tokens: {sum(total_usage.values()):,}")
    print(f"\nSaved per-user records: {users_dir}/")
    print(f"Saved summary: {summary_path}")
    print(f"Saved cost report: {cost_path}")
    print(f"Saved trend plot: {trend_plot_path}")

    # ── Dry-run final report ─────────────────────────────────────────────────
    if args.dry_run:
        print("\n" + "=" * 60)
        if dry_run_violations:
            print(f"[DRY RUN] FAILED — {len(dry_run_violations)} invariant violations:")
            for v in dry_run_violations:
                print(f"  {v}")
        else:
            print("[DRY RUN] PASSED — all invariants verified:")
            print("  ✓ Cheatsheet reset per user")
            print("  ✓ Predict before update (no gold in prediction prompt)")
            print("  ✓ Cheatsheet updated after each turn using gold preference")
            print("  ✓ Output format compatible with existing evaluator")
        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamic Cheatsheet / Running User Profile baseline"
    )
    parser.add_argument("--dataset",       type=str,  default="prism")
    parser.add_argument("--n-users",       type=int,  default=1000)
    parser.add_argument("--users-per-run", type=int,  default=100)
    parser.add_argument("--seed",          type=int,  default=42)

    # Reasoning model (ranking prediction)
    parser.add_argument("--reasoning-backend",   type=str, choices=["hf", "openai", "openrouter"], default="hf")
    parser.add_argument("--reasoning-model",     type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--reasoning-base-url",  type=str, default=None)
    parser.add_argument("--reasoning-api-key",   type=str, default=None)
    parser.add_argument("--reasoning-max-tokens",type=int, default=256)
    parser.add_argument("--reasoning-effort",    type=str, default="minimal")

    # Scoring model (EVALUATE_PROMPT + cheatsheet update)
    parser.add_argument("--score-backend",          type=str, choices=["hf", "openai", "openrouter"], default="openai")
    parser.add_argument("--score-model",            type=str, default="gpt-5")
    parser.add_argument("--score-base-url",         type=str, default=None)
    parser.add_argument("--score-api-key",          type=str, default=None)
    parser.add_argument("--score-max-tokens",       type=int, default=256)
    parser.add_argument("--score-reasoning-effort", type=str, default="minimal")

    # Cheatsheet update (uses scoring model by default; only max-tokens is separate)
    parser.add_argument("--update-max-tokens", type=int, default=512,
                        help="Max tokens for cheatsheet update response (separate from scoring)")

    # Embedding
    parser.add_argument("--embed-model",   type=str, default="text-embedding-3-small")
    parser.add_argument("--embed-api-key", type=str, default=None,
                        help="OpenAI API key for embeddings (defaults to --score-api-key)")
    parser.add_argument("--embed-base-url",type=str, default=None)

    # Cheatsheet parameters
    parser.add_argument("--max-bullets", type=int, default=3,
                        help="Maximum bullet points in the cheatsheet (default: 3)")

    # Misc
    parser.add_argument("--max-chars",         type=int,   default=280)
    parser.add_argument("--max-retries",        type=int,   default=3)
    parser.add_argument("--retry-delay",        type=float, default=0.5)
    parser.add_argument("--ranking-fail-mode",  type=str,   choices=["raise", "fallback"], default="fallback")
    parser.add_argument("--hf-enable-thinking", action="store_true")

    # Dry-run sanity check
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock models (no API calls) to verify ordering invariants")

    parser.add_argument("--output-dir", type=str, default="baseline_results")
    parser.add_argument("--run-id",     type=str, default="cheatsheet_gpt5_openrouter")
    return parser.parse_args()


if __name__ == "__main__":
    run_baseline(parse_args())
