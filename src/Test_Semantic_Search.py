import chromadb
import time
from chromadb.config import Settings
from embeddings import EmbeddingGenerator


def search_tickets(query_text, n_results=5, persist_directory="./chroma"):
    """Search for similar tickets in the vector database.

    Args:
        query_text: The search query text
        n_results: Number of results to return (default: 5)
        persist_directory: Path to ChromaDB persistence directory

    Returns:
        dict: Query results from ChromaDB with latency metrics
    """
    total_start = time.time()

    client = chromadb.Client(
        Settings(persist_directory=persist_directory, is_persistent=True)
    )

    collection = client.get_or_create_collection("tickets")
    print(f"Collection count: {collection.count()}")

    if collection:
        print("good !!")
    else:
        print("siir lhih o bki !!")

    # Embedding generation latency
    embed_start = time.time()
    embedder = EmbeddingGenerator()
    embeddings = embedder.encode(query_text)
    embed_time = (time.time() - embed_start) * 1000
    print(f"Embedding generation: {embed_time:.2f}ms")

    # Vector search latency
    query_start = time.time()
    results = collection.query(
        query_embeddings=embeddings.tolist(), n_results=n_results
    )
    query_time = (time.time() - query_start) * 1000
    print(f"Vector search: {query_time:.2f}ms")

    total_time = (time.time() - total_start) * 1000
    print(f"Total latency: {total_time:.2f}ms")

    response = {
        "results": results,
        "latency_ms": {
            "embedding": round(embed_time, 2),
            "vector_search": round(query_time, 2),
            "total": round(total_time, 2),
        },
    }

    return response


if __name__ == "__main__":
    result = search_tickets("server not responding")
    print(result)
