# utils.py - Text preprocessing utilities

import re
import logging
from typing import Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import ssl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix SSL for NLTK downloads on macOS
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception as e:
    logger.warning(f"SSL context initialization warning: {e}")

# Download required NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")
    raise

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def sanitize_text(text: str) -> str:
    """
    Sanitize and validate input text.
    
    Args:
        text: Raw text input
        
    Returns:
        Sanitized text
        
    Raises:
        ValueError: If text is empty or invalid after sanitization
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Remove HTML tags and excessive whitespace
    text = re.sub(r'<[^>]+>', '', text)
    text = ' '.join(text.split())
    
    if not text or len(text.strip()) == 0:
        raise ValueError("Text cannot be empty after sanitization")
    
    return text


def preprocess_text(text: str) -> str:
    """
    Preprocess raw text for model analysis.
    
    Performs the following operations:
    - Sanitizes input
    - Converts to lowercase
    - Removes special characters and numbers
    - Tokenizes into words
    - Removes English stopwords
    - Applies lemmatization
    
    Args:
        text: Raw article text to preprocess
        
    Returns:
        Cleaned and preprocessed text
        
    Raises:
        ValueError: If text is invalid or empty
        TypeError: If text is not a string
    """
    try:
        # Sanitize input
        text = sanitize_text(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers (keep only letters and spaces)
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize the text into words
        tokens = word_tokenize(text)
        
        # Remove stopwords and apply lemmatization
        cleaned_tokens = [
            lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in stop_words and len(word) > 1
        ]
        
        # Join tokens back into a string
        result = ' '.join(cleaned_tokens)
        
        if not result or len(result.split()) == 0:
            raise ValueError("No valid tokens found after preprocessing")
        
        return result
        
    except (ValueError, TypeError) as e:
        logger.error(f"Text preprocessing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise ValueError(f"Failed to preprocess text: {e}")
