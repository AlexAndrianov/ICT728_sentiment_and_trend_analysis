from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Optional, Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


DEFAULT_STOPWORDS = set(ENGLISH_STOP_WORDS).union(
    {
        "amp",
        "rt",
        "im",
        "ive",
        "dont",
        "didnt",
        "doesnt",
        "cant",
        "couldnt",
        "wouldnt",
        "shouldnt",
        "youre",
        "theyre",
        "thats",
        "lets",
        "us",
    }
)


def clean_post_text(text: str) -> str:
    """
    Normalize a social post for keyword extraction.

    - lowercase
    - remove urls
    - remove mentions
    - keep hashtag words but remove '#'
    - remove punctuation except apostrophes inside words
    - collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", "")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _valid_posts(posts: Sequence[str]) -> List[str]:
    cleaned = [clean_post_text(post) for post in posts if isinstance(post, str)]
    return [post for post in cleaned if post]


def _fallback_keywords(
    cleaned_posts: Sequence[str],
    max_words: int = 3,
    stopwords: Optional[set[str]] = None,
) -> str:
    stopwords = stopwords or DEFAULT_STOPWORDS
    counter: Counter[str] = Counter()

    for post in cleaned_posts:
        for token in post.split():
            if token not in stopwords and len(token) > 2 and not token.isdigit():
                counter[token] += 1

    top_tokens = [token for token, _ in counter.most_common(max_words)]
    if not top_tokens:
        return "Misc Trend"

    return " ".join(word.capitalize() for word in top_tokens)


def define_trend(
    posts: Sequence[str],
    max_words: int = 3,
    min_df: int = 1,
    max_features: int = 1000,
) -> str:
    """
    Create a short trend label (1-3 words) from a group of related posts.

    Strategy:
    1. Clean all posts.
    2. Use TF-IDF over unigrams + bigrams.
    3. Pick the highest-scoring term across the cluster.
    4. Convert it into a short title.

    Returns:
        A trend label like:
        - "Climate Change"
        - "Python Tutorials"
        - "AI Tools"
        - "Misc Trend" for empty/invalid input
    """
    cleaned_posts = _valid_posts(posts)
    if not cleaned_posts:
        return "Misc Trend"

    # If there is too little text, use a simpler fallback.
    if len(" ".join(cleaned_posts).split()) < 3:
        return _fallback_keywords(cleaned_posts, max_words=max_words)

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=min_df,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9']+\b",
        )
        matrix = vectorizer.fit_transform(cleaned_posts)
        feature_names = vectorizer.get_feature_names_out()

        # Sum scores across all posts in the same cluster
        scores = matrix.sum(axis=0).A1
        ranked_indices = scores.argsort()[::-1]

        for idx in ranked_indices:
            candidate = feature_names[idx].strip()
            if not candidate:
                continue

            words = [
                w for w in candidate.split()
                if w not in DEFAULT_STOPWORDS and len(w) > 2
            ]
            if not words:
                continue

            words = words[:max_words]
            return " ".join(word.capitalize() for word in words)

    except ValueError:
        # Example: all posts become empty after preprocessing
        pass

    return _fallback_keywords(cleaned_posts, max_words=max_words)


def define_trends_for_clusters(
    posts: Sequence[str],
    cluster_labels: Sequence[int],
    max_words: int = 3,
) -> dict[int, str]:
    """
    Build one trend label per cluster.

    Args:
        posts: original posts
        cluster_labels: cluster index for each post

    Returns:
        dict like {0: "Football Finals", 1: "Climate Change"}
    """
    if len(posts) != len(cluster_labels):
        raise ValueError("posts and cluster_labels must have the same length")

    grouped_posts: dict[int, List[str]] = {}
    for post, label in zip(posts, cluster_labels):
        grouped_posts.setdefault(int(label), []).append(post)

    return {
        cluster_id: define_trend(cluster_posts, max_words=max_words)
        for cluster_id, cluster_posts in grouped_posts.items()
    }


def match_existing_trend(
    trend_name: str,
    existing_trends: Iterable[str],
    threshold: float = 0.6,
) -> str:
    """
    Reuse an existing trend name if it is sufficiently similar.

    Similarity is based on token overlap:
        score = intersection / max(token_count_a, token_count_b)

    This avoids adding new dependencies.
    """
    normalized_new = set(clean_post_text(trend_name).split())
    if not normalized_new:
        return trend_name

    best_name = trend_name
    best_score = 0.0

    for existing in existing_trends:
        normalized_existing = set(clean_post_text(existing).split())
        if not normalized_existing:
            continue

        overlap = len(normalized_new & normalized_existing)
        score = overlap / max(len(normalized_new), len(normalized_existing))

        if score > best_score:
            best_score = score
            best_name = existing

    return best_name if best_score >= threshold else trend_name