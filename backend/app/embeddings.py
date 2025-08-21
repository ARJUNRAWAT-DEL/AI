# Thin wrapper to reuse the ai_utils singleton embedder
from typing import List
from .ai_utils import generate_embedding

def get_embedding(text: str) -> List[float]:
    return generate_embedding(text)
