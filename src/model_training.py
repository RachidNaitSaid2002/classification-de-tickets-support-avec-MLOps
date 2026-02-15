from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

from src.load_data import load_data_from_chroma

def train_model(X, y):

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

    # Save model + encoder
    joblib.dump(clf, "model.joblib")
    joblib.dump(encoder, "label_encoder.joblib")

    print("Model saved successfully.")

    return clf

if __name__ == "__main__":
    Features, labels = load_data_from_chroma()
    train_model(Features, labels)

