from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def extract_features(corpus, max_features=5000):
    """Generate TF-IDF features."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def save_vectorizer(vectorizer, path):
    joblib.dump(vectorizer, path)

def load_vectorizer(path):
    return joblib.load(path)
