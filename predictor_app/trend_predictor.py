from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterable

from sentence_transformers import SentenceTransformer

_BUNDLE_LOCK = threading.Lock()
_BUNDLES: dict[str, dict] = {}

logger = logging.getLogger(__name__)

_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER = None
_KEYBERT_LOCK = threading.Lock()
_KEYBERT_MODEL = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    with _EMBEDDER_LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER

        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        return _EMBEDDER


def _get_keybert():
    global _KEYBERT_MODEL
    if _KEYBERT_MODEL is not None:
        return _KEYBERT_MODEL

    with _KEYBERT_LOCK:
        if _KEYBERT_MODEL is not None:
            return _KEYBERT_MODEL

        from keybert import KeyBERT

        _KEYBERT_MODEL = KeyBERT(model=_get_embedder())
        return _KEYBERT_MODEL


def _artifact_path() -> Path:
    return Path(__file__).resolve().parent / "ml_artifacts" / "ridge_views.joblib"


def _xgb_artifact_path() -> Path:
    return Path(__file__).resolve().parent / "ml_artifacts" / "xgb_views.joblib"


def _artifact_path_for_model(model_key: str) -> Path:
    if model_key == "xgb":
        return _xgb_artifact_path()
    elif model_key == "ridge":
        return _artifact_path()

    raise ValueError(f"Unknown model key: {model_key}")


def _load_bundle(model_key: str) -> dict:
    cached = _BUNDLES.get(model_key)
    if cached is not None:
        return cached

    with _BUNDLE_LOCK:
        cached = _BUNDLES.get(model_key)
        if cached is not None:
            return cached

        import joblib

        path = _artifact_path_for_model(model_key)
        if not path.exists():
            raise FileNotFoundError(f"Model artifacts not found for '{model_key}': {path}")

        raw = joblib.load(path)
        if not isinstance(raw, dict):
            raise TypeError(f"Invalid artifact format for '{model_key}': expected dict bundle, got {type(raw)}")

        bundle = dict(raw)
        if "model" not in bundle:
            raise ValueError(f"Invalid artifact bundle for '{model_key}': missing required key 'model'")

        return bundle


def _num_feature_value(tweet, col: str) -> float:
    # Map notebook feature names -> TweetPost model fields
    mapping = {
        "date_posted": "posted_hour",
        "photos": "photos_count",
        "videos": "videos_count",
        "hashtags": "hashtags_count",
        "followers": "followers_count",
        "posts_count": "posts_count",
        "following": "following_count",
        "is_verified": "is_verified",
        "tagged_users": "tagged_users_count",
        "description_error_count": "description_error_count",
        "swear_word_count": "swear_word_count",
        "word_count": "word_count",
        "emoji_count": "emoji_count",
        "uppercase_word_count": "uppercase_word_count",
    }

    field = mapping.get(col)
    if not field:
        return 0.0

    val = getattr(tweet, field, 0)
    if val is None:
        return 0.0

    if field == "is_verified":
        return 1.0 if bool(val) else 0.0

    try:
        return float(val)
    except Exception:
        return 0.0


def _get_selected_model_key() -> str:
    try:
        from .models import TrendModelSettings

        obj = TrendModelSettings.objects.order_by("id").first()
        if obj and obj.selected_model:
            return str(obj.selected_model)
    except Exception:
        pass
    return "ridge"


def predict_views_for_tweet(tweet) -> int:
    """Return predicted views for a single tweet.
    """

    import numpy as np

    model_key = _get_selected_model_key()
    logger.info(
        "predict_views_for_tweet: start tweet_id=%s model_key=%s",
        getattr(tweet, "id", None),
        model_key,
    )

    bundle = _load_bundle(model_key)
    logger.info(
        "predict_views_for_tweet: bundle loaded keys=%s",
        sorted(list(bundle.keys())),
    )

    model = bundle.get("model")
    num_cols = bundle.get("num_cols")
    text_scaler = bundle.get("text_scaler")

    logger.info(
        "predict_views_for_tweet: model=%s num_cols=%s text_scaler=%s",
        type(model).__name__ if model is not None else None,
        len(num_cols) if num_cols is not None else None,
        type(text_scaler).__name__ if text_scaler is not None else None,
    )

    X_num = np.array(
        [[_num_feature_value(tweet, c) for c in num_cols]],
        dtype=np.float32,
    )

    logger.info(
        "predict_views_for_tweet: X_num shape=%s sample=%s",
        getattr(X_num, "shape", None),
        X_num[0][: min(5, X_num.shape[1])].tolist() if getattr(X_num, "ndim", 0) == 2 else None,
    )

    texts = [(tweet.content or "").strip()]
    logger.info(
        "predict_views_for_tweet: text_len=%s",
        len(texts[0]) if texts else 0,
    )

    text_model = _get_embedder()
    try:
        X_text = text_model.encode(texts, batch_size=32, show_progress_bar=False)
        logger.info(
            "predict_views_for_tweet: X_text shape=%s",
            getattr(X_text, "shape", None),
        )
    except Exception:
        logger.exception("predict_views_for_tweet: text encode failed")
        raise

    try:
        X_text = text_scaler.transform(X_text)
        logger.info(
            "predict_views_for_tweet: X_text_scaled shape=%s",
            getattr(X_text, "shape", None),
        )
    except Exception:
        logger.exception(
            "predict_views_for_tweet: text_scaler.transform failed (text_scaler=%s)",
            type(text_scaler).__name__ if text_scaler is not None else None,
        )
        raise

    X = np.hstack([X_text, X_num])
    logger.info(
        "predict_views_for_tweet: X shape=%s",
        getattr(X, "shape", None),
    )

    try:
        y_pred = model.predict(X)
        logger.info("predict_views_for_tweet: y_pred raw=%s", y_pred)
    except Exception:
        logger.exception("predict_views_for_tweet: model.predict failed (model=%s)", type(model).__name__)
        raise

    try:
        y_pred = np.expm1(y_pred)
        logger.info("predict_views_for_tweet: y_pred expm1=%s", y_pred)
    except Exception:
        logger.exception("predict_views_for_tweet: np.expm1 failed")
        raise

    try:
        out = float(y_pred[0])
        logger.info("predict_views_for_tweet: y_pred[0]=%s", out)
        return out
    except Exception:
        logger.exception("predict_views_for_tweet: failed to read y_pred[0]")
        return 0


def cluster_by_trends(tweets: list[str], distance_threshold: float = 0.35) -> dict[int, int]:
    """Cluster tweets by semantic meaning.

    Args:
        tweets: List of tweet texts.
        distance_threshold: Cosine distance threshold for agglomerative clustering.

    Returns:
        Mapping {tweet_index: cluster_index}.
    """
    if not tweets:
        return {}

    cleaned = [(t or "").strip() for t in tweets]
    non_empty_indices = [i for i, text in enumerate(cleaned) if text]

    if not non_empty_indices:
        return {i: i for i in range(len(tweets))}

    text_model = _get_embedder()
    embeddings = text_model.encode(
        [cleaned[i] for i in non_empty_indices],
        show_progress_bar=False,
        convert_to_tensor=True,
    )

    # Popular ready-to-use semantic clustering from sentence-transformers.
    from sentence_transformers.util import community_detection

    similarity_threshold = max(0.0, min(1.0, 1.0 - float(distance_threshold)))
    communities = community_detection(
        embeddings=embeddings,
        threshold=similarity_threshold,
        min_community_size=1,
    )

    result: dict[int, int] = {}
    for cluster_idx, community in enumerate(communities):
        for local_idx in community:
            tweet_idx = non_empty_indices[int(local_idx)]
            result[tweet_idx] = cluster_idx

    # Put empty tweets in separate singleton clusters.
    next_cluster = (max(result.values()) + 1) if result else 0
    for idx, text in enumerate(cleaned):
        if not text:
            result[idx] = next_cluster
            next_cluster += 1

    return result


def define_cluster_name(tweet_to_cluster: dict[int, int], tweets: list[str]) -> dict[int, str]:
    """Generate short 1-3 word names for each discovered cluster.

    Args:
        tweet_to_cluster: Mapping {tweet_index: cluster_index}.
        tweets: List of tweet texts.

    Returns:
        Mapping {cluster_index: cluster_name}.
    """
    if not tweet_to_cluster:
        return {}

    from collections import Counter, defaultdict
    import re

    cluster_to_tweets = defaultdict(list)
    for tweet_idx, cluster_idx in tweet_to_cluster.items():
        if 0 <= tweet_idx < len(tweets):
            text = (tweets[tweet_idx] or "").strip()
            if text:
                cluster_to_tweets[int(cluster_idx)].append(text)

    cluster_names: dict[int, str] = {}

    for cluster_idx, texts in cluster_to_tweets.items():
        if not texts:
            cluster_names[cluster_idx] = ""
            continue

        try:
            kw_model = _get_keybert()
            merged_text = " ".join(texts)

            keywords = kw_model.extract_keywords(
                merged_text,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                use_mmr=True,
                diversity=0.6,
                top_n=12,
            )

            selected_name = ""
            for phrase, score in keywords:
                candidate = phrase.strip().lower()
                words = candidate.split()

                if score < 0.2:
                    continue

                if not (1 <= len(words) <= 3):
                    continue

                selected_name = " ".join(words)
                break

            if selected_name:
                cluster_names[cluster_idx] = selected_name
                continue

            # Fallback from raw cluster text, still without synthetic labels.
            fallback_tokens = re.findall(r"[a-zA-Z]{3,}", merged_text.lower())
            if fallback_tokens:
                common = [w for w, _ in Counter(fallback_tokens).most_common(3)]
                cluster_names[cluster_idx] = " ".join(common)
            else:
                cluster_names[cluster_idx] = ""
        except Exception:
            logger.exception("define_cluster_name: failed for cluster %s", cluster_idx)
            merged_text = " ".join(texts)
            fallback_tokens = re.findall(r"[a-zA-Z]{3,}", merged_text.lower())
            if fallback_tokens:
                common = [w for w, _ in Counter(fallback_tokens).most_common(3)]
                cluster_names[cluster_idx] = " ".join(common)
            else:
                cluster_names[cluster_idx] = ""

    return cluster_names


def define_similar_cluster(
    cluster_names: dict[int, str],
    previous_cluster_names: Iterable[str],
    similarity_threshold: float = 0.7,
) -> dict[int, str]:
    """Replace new cluster names with semantically similar old names.

    Args:
        cluster_names: Mapping {cluster_index: generated_cluster_name}.
        previous_cluster_names: Past cluster names from previous iterations.
        similarity_threshold: Minimum cosine similarity to reuse an old name.

    Returns:
        Mapping {cluster_index: verified_cluster_name}.
    """
    if not cluster_names:
        return {}

    old_names = [str(name).strip() for name in previous_cluster_names if str(name).strip()]
    if not old_names:
        return dict(cluster_names)

    text_model = _get_embedder()
    old_embeddings = text_model.encode(old_names, show_progress_bar=False, normalize_embeddings=True)

    import numpy as np

    verified = {}
    for cluster_idx, new_name in cluster_names.items():
        current_name = (new_name or "").strip()
        if not current_name:
            verified[cluster_idx] = new_name
            continue

        new_embedding = text_model.encode([current_name], show_progress_bar=False, normalize_embeddings=True)[0]
        similarities = np.dot(old_embeddings, new_embedding)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= similarity_threshold:
            verified[cluster_idx] = old_names[best_idx]
        else:
            verified[cluster_idx] = current_name

    return verified
