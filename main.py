import os
import pandas as pd
import string
import joblib
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -------------------
# Text Preprocessing
# -------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text, preserve_line=True)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# -------------------
# Main Pipeline
# -------------------
def main():
    data_path = r"C:\Users\ragul\OneDrive\Desktop\Bias Detection in AI-Generated\data\raw\bias_detection_combined_train.csv"
    
    print("Loading and cleaning dataset...")
    df = pd.read_csv(data_path)

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    df['cleaned_text'] = df['text'].astype(str).apply(clean_text)

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df['label'])

    print("Extracting features using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['encoded_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_.astype(str)))

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "bias_detector.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))

    print("Training completed and models saved successfully.")

if __name__ == "__main__":
    main()
