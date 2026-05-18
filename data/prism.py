from typing import Dict, List
from datasets import load_dataset
from .base import Turn, Conversation, UserData
import random

MIN_USER_TURNS = 20

def group_by_turns(conversation_history: List[Dict], id_prefix: str = "") -> List[Turn]:
    turns = {}
    for msg in conversation_history:
        turn = msg.get('turn', 0)
        if turn not in turns:
            turns[turn] = Turn()
        role = msg.get('role')
        if role == 'user':
            turns[turn].user_message = msg.get('content')
        else:
            turns[turn].candidates.append(msg.get('content'))
            if msg.get('if_chosen', False):
                turns[turn].chosen = msg.get('content')
    return [
        turns[i] for i in sorted(turns.keys())
        if turns[i].user_message != "EMPTY STRING"
    ]

def extract_profile(survey: dict) -> str:
    key_fields = [
        'age',
        'gender',
        'religion',
        'ethnicity',
        'self_description',
        'system_string',
    ]
    profile = "\n".join([f"{field.replace('_', ' ').title()}: {survey[field]}" for field in key_fields if field in survey and survey[field]])
    stated_prefs: dict = survey["stated_prefs"]
    stated_prefs.pop("other")
    stated_prefs.pop("other_text")
    mean_score = sum(stated_prefs.values()) / len(stated_prefs)
    prior_prefs = [k for k, v in stated_prefs.items() if v > mean_score]
    low_prefs = [k for k, v in stated_prefs.items() if v <= mean_score]
    profile += "\nPrioritized aspects: " + ", ".join(prior_prefs)
    profile += "\nComparatively less prioritzed aspects: " + ", ".join(low_prefs)
    return profile

def load_prism(n_users: int = None, seed: int = None) -> List[UserData]:
    """
    Load PRISM conversations and return per-user bundles with normalized conversations and turns.
    If n_users is provided, limit to the first n unique users (by dataset order).
    """
    train_data = load_dataset("HannahRoseKirk/prism-alignment", "conversations")['train']

    user_conversations: Dict[List[Dict]] = {}
    user_order: List[str] = []
    for rec in train_data:
        uid = rec['user_id']
        cid = rec.get('conversation_id', 'unknown')
        conv_hist = rec.get('conversation_history', None)
        if uid not in user_conversations:
            user_conversations[uid] = []
            user_order.append(uid)
        user_conversations[uid].append({
            'conversation_id': cid,
            'conversation_history': conv_hist
        })

    survey_data = load_dataset("HannahRoseKirk/prism-alignment", "survey")['train']
    survey_rec = {rec['user_id']: rec for rec in survey_data}

    # --- Build all users in dataset order, apply MIN_USER_TURNS filter ---
    # Matches preference-tracing-v2/data/prism.py exactly.
    all_users: List[UserData] = []
    skipped_short = 0
    for uid in user_order:
        convs = []
        for conv in user_conversations.get(uid, []):
            cid = conv['conversation_id']
            history = conv['conversation_history']
            turns = group_by_turns(history)
            convs.append(Conversation(
                conversation_id=cid,
                turns=turns
            ))
        total_turns = sum(len(conv.turns) for conv in convs)
        if total_turns < MIN_USER_TURNS:
            skipped_short += 1
            continue
        gt_profile = extract_profile(survey_rec[uid])
        all_users.append(UserData(
            user_id=uid,
            conversations=convs,
            gt_profile=gt_profile
        ))

    print(
        f"[Data] Total users in dataset: {len(user_order)}  |  "
        f"Skipped (turns < {MIN_USER_TURNS}): {skipped_short}  |  "
        f"Eligible: {len(all_users)}"
    )

    # --- Sample only if eligible > n_users (matches v2 logic exactly) ---
    # With n_users=1000 and ~619 eligible, condition is False → return all 619
    # in dataset order (no shuffle).  The first users_per_run entries are
    # therefore the first N eligible users in the original dataset order,
    # which is what v2 does and what produces the expected user_ids.
    if n_users is not None and len(all_users) > n_users:
        random.seed(seed)
        users = random.sample(all_users, n_users)
    else:
        users = all_users

    print(
        f"[Data] Requested n_users={n_users}  |  "
        f"Final pool: {len(users)}  |  "
        f"Avg turns/user: {sum(sum(len(c.turns) for c in u.conversations) for u in users) / len(users):.1f}"
    )
    return users


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick test for PRISM adapter output")
    parser.add_argument("--n-users", type=int, default=2, help="Number of users to load for preview")
    parser.add_argument("--preview", action="store_true", help="Print a short preview of the first user's first conversation")
    parser.add_argument("--dump-json", action="store_true", help="Dump the first user's bundle as JSON (truncated)")
    args = parser.parse_args()

    users = load_prism(n_users=args.n_users)
    print(users)