from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

class MTEBModel(ABC):
    @abstractmethod
    def encode(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of sentences into embeddings.
        """
        pass

class SentenceTransformerModel(MTEBModel):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(sentences, batch_size=batch_size, show_progress_bar=True)
