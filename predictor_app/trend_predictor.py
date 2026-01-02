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


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    with _EMBEDDER_LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER

        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        return _EMBEDDER


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
