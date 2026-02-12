from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name="intfloat/multilingual-e5-base"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        # Add prefix for E5 model
        texts = ["passage: " + text for text in texts]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings
