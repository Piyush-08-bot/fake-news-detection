# utils.py - all our preprocessing functions are here

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# fix SSL for NLTK downloads on macOS
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

# downloading required nltk data
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# setting up lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """this function takes raw text and returns clean preprocessed text"""

    # converting to lowercase
    text = text.lower()

    # removing special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # tokenizing the text into words
    tokens = word_tokenize(text)

    # removing stopwords and applying lemmatization
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # joining tokens back into a string
    return ' '.join(cleaned_tokens)
