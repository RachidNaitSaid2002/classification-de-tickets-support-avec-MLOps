import chromadb
import numpy as np
from chromadb.config import Settings


def load_data_from_chroma(collection_name="tickets"):
    client = chromadb.Client(
        Settings(
            persist_directory="./chroma",
            is_persistent=True
        )
    )
    collection = client.get_collection(collection_name)
    data = collection.get(include=["embeddings", "metadatas", "documents"])
    embeddings = np.array(data["embeddings"])
    labels = [meta["type"] for meta in data["metadatas"]]
    return embeddings, labels


if __name__ == "__main__":
    embd, labels = load_data_from_chroma()
    print(embd[0])
    print(labels[0])
