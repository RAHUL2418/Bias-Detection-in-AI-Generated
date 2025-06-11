import re
import nltk
from nltk.corpus import stopwords

# Download stopwords only if not already downloaded
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """
    Clean and normalize input text by:
    - converting to lowercase
    - removing URLs
    - removing non-alphabetic characters
    - removing stopwords

    Args:
        text (str): Input text string.

    Returns:
        str: Cleaned and normalized text.
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)                # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)            # Remove punctuation and numbers
    words = [word for word in text.split() if word not in STOPWORDS]
    cleaned_text = " ".join(words)
    return cleaned_text
