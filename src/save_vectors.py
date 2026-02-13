import chromadb


class VectorStore:
    def __init__(self, collection_name="tickets", persist_directory="chroma"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_embeddings(self, ids, embeddings, documents, batch_size=5000):
        embeddings_list = embeddings.tolist()
        total = len(ids)

        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            print(f"Adding batch {i // batch_size + 1}: items {i} to {end_idx - 1}")

            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings_list[i:end_idx],
                documents=documents[i:end_idx],
            )
        print(f"Successfully added {total} embeddings")
