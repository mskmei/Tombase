from dataclasses import dataclass, field
from typing import List

@dataclass
class Turn:
    turn: int = 0
    user_message: str = ""
    candidates: List[str] = field(default_factory=list)
    chosen: str = ""
    chosen_idx: int = -1
    
    def __post_init__(self):
        if self.chosen_idx >= 0:
            if self.chosen_idx >= len(self.candidates):
                raise ValueError(f"chosen_idx {self.chosen_idx} out of range for candidates list of length {len(self.candidates)}")
            if self.candidates[self.chosen_idx] != self.chosen:
                raise ValueError(f"chosen candidate '{self.chosen}' does not match expected candidate at index {self.chosen_idx}: '{self.candidates[self.chosen_idx]}'")
        elif self.chosen and self.candidates:
            if self.chosen in self.candidates:
                self.chosen_idx = self.candidates.index(self.chosen)
            else:
                raise ValueError(f"chosen candidate '{self.chosen}' not found in candidates list: {self.candidates}")
    
    def __repr__(self):
        user_preview = self.user_message[:100] + ("..." if len(self.user_message) > 100 else "")
        candidate_lines = "\n".join(
            [c[:100] + ("..." if len(c) > 100 else "") for c in self.candidates]
        )
        chosen_preview = self.chosen[:100] + ("..." if len(self.chosen) > 100 else "")
        return (
            f"Turn {self.turn}:\n"
            f"User Message:\n {user_preview}\n\n"
            f"Candidates:\n{candidate_lines}\n\n"
            f"Chosen:\n{chosen_preview}\n"
        )
    
    def format(self, include_candidates: bool = True, include_choice: bool = True) -> str:
        formatted = f"User: {self.user_message}\n"
        if include_candidates:
            formatted += "Candidates:\n"
            for idx, cand in enumerate(self.candidates):
                marker = "[CHOSEN]" if idx == self.chosen_idx else "[REJECTED]"
                formatted += f"{marker} {idx+1}. {cand}\n" if include_choice else f"{idx+1}. {cand}\n"
        else:
            formatted += f"Model: {self.chosen}\n"
        return formatted
    
@dataclass
class Conversation:
    conversation_id: str = ""
    turns: List[Turn] = field(default_factory=list)
    
    def __repr__(self):
        turns_text = "\n\n".join([repr(turn) for turn in self.turns])
        return f"Conversation {self.conversation_id}:\n{turns_text}\n"
    
@dataclass
class UserData:
    user_id: str = ""
    conversations: List[Conversation] = field(default_factory=list)
    gt_profile: str = ""
    
    def __repr__(self):
        conv_text = "\n\n".join([repr(conv) for conv in self.conversations])
        return f"User {self.user_id} ({self.gt_profile}):\n\n{conv_text}\n"