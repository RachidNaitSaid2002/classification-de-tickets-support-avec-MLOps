import joblib
from embeddings import EmbeddingGenerator

# Load model
model = joblib.load("models/model.joblib")

# Generate embedding for test text
embedder = EmbeddingGenerator()
test_embedding = embedder.encode(["This is a test ticket"])

# Predict
prediction = model.predict(test_embedding)
print(f"Predicted class: {prediction[0]}")
