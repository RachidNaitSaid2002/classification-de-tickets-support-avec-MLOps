import chromadb


class VectorStore:
    def __init__(self, collection_name="tickets", persist_directory="chroma"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_embeddings(self, ids, embeddings, documents, labels=None, batch_size=5000):
        embeddings_list = embeddings.tolist()
        total = len(ids)

        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            print(f"Adding batch {i // batch_size + 1}: items {i} to {end_idx - 1}")

            batch_data = {
                "ids": ids[i:end_idx],
                "embeddings": embeddings_list[i:end_idx],
                "documents": documents[i:end_idx],
            }

            if labels is not None:
                batch_data["metadatas"] = [
                    {"type": str(label)} for label in labels[i:end_idx]
                ]

            self.collection.add(**batch_data)

        print(f"Successfully added {total} embeddings")
