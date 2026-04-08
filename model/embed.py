import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class EmbedConfig:
    backend: str = "openai"
    model: str = "text-embedding-3-small"
    dim: int = 1536
    api_key: str = None
    base_url: str = "https://api.openai.com/v1"


_client = None

def get_client(api_key: str = None, base_url: str = "https://api.openai.com/v1"):
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client

def embed(
    text: str | List[str],
    *,
    embed_cfg: EmbedConfig
) -> np.ndarray:
    """
    Embed text into vectors using OpenAI API.

    Args:
        text: String or list of strings to embed.
        embed_cfg: Configuration for the embedding model.

    Returns:
        2D numpy array of shape (N, D), normalized.
    """
    if isinstance(text, str):
        text = [text]
    
    client = get_client(embed_cfg.api_key, embed_cfg.base_url)
    response = client.embeddings.create(
        model=embed_cfg.model,
        input=text
    )
    vec = np.array([d.embedding for d in sorted(response.data, key=lambda x: x.index)], dtype=np.float32)
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-14)
    return vec


def text_similarity(text1: str, text2: str, embed_cfg: EmbedConfig) -> float:
    """Compute cosine similarity between two texts."""
    vec = embed([text1, text2], embed_cfg=embed_cfg)
    return float(np.dot(vec[0], vec[1]).item())


def relative_similarity_score(adapted: str, candidates: List[str], chosen_idx: int, embed_cfg: EmbedConfig) -> float:
    """Compute relative similarity: similarity to chosen minus max similarity to rejected."""
    vec = embed([adapted] + candidates, embed_cfg=embed_cfg)
    adapted_vec = vec[0]
    candidate_vecs = vec[1:]
    similarities = np.dot(candidate_vecs, adapted_vec)
    chosen_similarity = similarities[chosen_idx]
    rejected_similarities = np.delete(similarities, chosen_idx)
    return float((chosen_similarity - np.max(rejected_similarities)).item())
