import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("predictor_app")

# Import NLTK Vader
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logger.info("Downloading NLTK vader_lexicon...")
    nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()
logger.info("NLTK Vader sentiment analyzer initialized successfully")


def _clip(s: str, n: int = 240) -> str:
    try:
        s2 = str(s)
    except Exception:
        return "<unprintable>"
    if len(s2) <= n:
        return s2
    return s2[:n] + "…"


@dataclass(frozen=True)
class SentimentResult:
    # -1 negative, 0 neutral/unknown, 1 positive
    score: int
    label: str


def _clean_text_simple(text: str) -> str:
    """Simple text cleaning for sentiment analysis"""
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,6}\b', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def vader_sentiment(text: str) -> SentimentResult:
    """Use NLTK VADER for sentiment analysis"""
    # Get sentiment scores
    scores = sia.polarity_scores(text)
    
    # Log scores for debugging
    logger.info(f"VADER scores for '{_clip(text)}': {scores}")
    
    # Determine sentiment based on compound score
    compound = scores['compound']
    
    # Standard VADER thresholds
    if compound >= 0.33333:
        return SentimentResult(score=1, label="positive")
    elif compound <= -0.33333:
        return SentimentResult(score=-1, label="negative")
    else:
        return SentimentResult(score=0, label="neutral")


def predict_sentiment(text: str, model_id: int) -> SentimentResult:
    """
    Predict sentiment using NLTK VADER
    
    Args:
        text: Input text to analyze
        model_id: Model ID (ignored, always uses VADER)
    
    Returns:
        SentimentResult with score (-1, 0, 1) and label
    """
    raw_text = str(text or "")
    if not raw_text.strip():
        return SentimentResult(score=0, label="neutral")
    
    # Clean text
    clean_text = _clean_text_simple(raw_text)
    
    logger.info(
        "sentiment: model_id=%s text_len=%s clean_len=%s raw=%r clean=%r",
        model_id,
        len(raw_text),
        len(clean_text),
        _clip(raw_text),
        _clip(clean_text),
    )
    
    # Always use NLTK VADER
    result = vader_sentiment(clean_text)
    logger.info(f"VADER result: score={result.score}, label={result.label}")
    return result
