from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingGenerator:
    def __init__(self, model_name="intfloat/multilingual-e5-base"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        # Convert to string and handle NaN/float values
        texts = [str(text) if text is not None else "" for text in texts]
        # Add prefix for E5 model
        texts = ["passage: " + text for text in texts]

        embeddings = self.model.encode(
            texts, show_progress_bar=True, normalize_embeddings=True
        )
        return embeddings
