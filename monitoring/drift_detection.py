import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.load_data import load_data_from_chroma

# Load dataset
Features, labels = load_data_from_chroma()


# Extract statistics from embedding vectors
def extract_vector_stats(vectors):
    vectors = np.array(vectors)
    return pd.DataFrame(
        {
            "vec_mean": vectors.mean(axis=1),
            "vec_std": vectors.std(axis=1),
            "vec_min": vectors.min(axis=1),
            "vec_max": vectors.max(axis=1),
            "vec_norm": np.linalg.norm(vectors, axis=1),
        }
    )


df = extract_vector_stats(Features)
df["labels"] = labels

# Load model and encoder
model = joblib.load("models/model.joblib")
encoder = joblib.load("models/encoder.joblib")

# Add predictions and encoded labels
df["prediction"] = model.predict(Features)
df["label_encoded"] = encoder.transform(df["labels"])

# Split the data to simulate "past" and "present"
reference_df = df.sample(frac=0.7, random_state=42)
current_df = df.drop(reference_df.index)

# Define the data mapping using vector statistics
data_definition = DataDefinition(
    numerical_columns=["vec_mean", "vec_std", "vec_min", "vec_max", "vec_norm"],
    categorical_columns=["labels", "prediction", "label_encoded"],
)

# Wrap data in Evidently Dataset objects
reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
current_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)

# Create a Report with the Data Drift preset
report = Report(metrics=[DataDriftPreset()])

# Run the report
result = report.run(reference_data=reference_dataset, current_data=current_dataset)

# Save it as an interactive HTML file
result.save_html(str(Path(__file__).parent / "drift_report.html"))
