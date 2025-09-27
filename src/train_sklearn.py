import os, joblib, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from data_loader import load_movie_reviews
from preprocess import clean_text

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    texts, labels = load_movie_reviews()
    X = [clean_text(t) for t in texts]
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000)),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    print("=== Logistic Regression ===")
    print(classification_report(y_test, preds, digits=4))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    joblib.dump(pipe, os.path.join(MODELS_DIR, "tfidf_logreg.joblib"))
    print("\nSaved: models/tfidf_logreg.joblib")

if __name__ == "__main__":
    main()
