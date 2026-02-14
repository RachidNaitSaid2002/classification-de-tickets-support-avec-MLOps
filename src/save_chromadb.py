from embeddings import EmbeddingGenerator
from save_vectors import VectorStore
import pandas as pd

df = pd.read_csv("data/processed/clean3.csv")
df = df.dropna(subset=["text_final"])
embedder = EmbeddingGenerator()
embeddings = embedder.encode(df["text_final"].tolist())

vector_store = VectorStore(persist_directory="chroma")

vector_store.add_embeddings(
    ids=[str(i) for i in range(len(df))],
    embeddings=embeddings,
    documents=df["text_final"].tolist(),
    labels=df["type"].tolist(),
)
