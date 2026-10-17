"""
Text Preprocessing Module.
Applies consistent lowercasing, punctuation/digit removal, tokenization,
stopword filtering, and WordNet lemmatization.
"""
import re
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """
    Standardized NLP Preprocessor implementing Functional Requirement 4.1.
    Ensures identical cleaning steps for queries and documents.
    """

    def __init__(self):
        self._ensure_nltk_resources()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def _ensure_nltk_resources(self):
        """Ensures required NLTK corpora are downloaded."""
        resources = [
            ("tokenizers/punkt", "punkt"),
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
            ("tokenizers/punkt_tab", "punkt_tab"),
        ]
        for path, name in resources:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(name, quiet=True)

    def preprocess(self, text: str) -> List[str]:
        """
        Cleans, lowercases, tokenizes, removes stopwords, and lemmatizes input text.
        
        Args:
            text: Raw input string
            
        Returns:
            List of filtered and lemmatized string tokens
        """
        if not isinstance(text, str):
            return []

        # Step A: Lowercase
        text = text.lower()

        # Step B: Remove punctuation and numbers
        text = re.sub(r"[^a-z\s]", " ", text)

        # Step C: Tokenize
        tokens = word_tokenize(text)

        # Step D: Remove stopwords and short tokens, then lemmatize
        filtered_tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 1
        ]

        return filtered_tokens

    def preprocess_with_string(self, text: str) -> Tuple[List[str], str]:
        """
        Returns both the token list and the space-joined string for TF-IDF.
        """
        tokens = self.preprocess(text)
        text_str = " ".join(tokens)
        return tokens, text_str


# Global reusable instance
_preprocessor_instance = TextPreprocessor()


def preprocess_text(text: str) -> List[str]:
    """Convenience functional wrapper for text preprocessing."""
    return _preprocessor_instance.preprocess(text)


def preprocess_query(text: str) -> Tuple[List[str], str]:
    """Convenience wrapper returning (tokens, joined_string)."""
    return _preprocessor_instance.preprocess_with_string(text)
