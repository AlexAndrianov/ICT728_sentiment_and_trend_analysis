from __future__ import annotations

import logging
import pickle
import re
import string
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

logger = logging.getLogger("predictor_app")


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


MODEL_CHOICES = [
    (1, "Naive_Bayes_Model_With_Simple_Tokenizer"),
    (2, "Stacking_Classifier_Logistic_Regression_Plus_SVC"),
    (3, "PyTorch_Neural_Network_Model"),
]


# Caches
_NB_MODEL = None
_SK_MODEL = None
_PT_MODEL = None
_PT_VOCAB = None
_PT_LOCK = threading.Lock()
_PT_WARMED = False


def _artifact_dir() -> Path:
    return Path(__file__).resolve().parent / "ml_artifacts" / "sentiment_analysis"


def _nb_path() -> Path:
    return _artifact_dir() / "naive_bayes_model_with_simple_tokenizer.pkl"


def _sk_path() -> Path:
    return _artifact_dir() / "StackingClassifierLogisticRegressionPlusSVC.pkl"


def _pt_path() -> Path:
    return _artifact_dir() / "PyTorchNBoWModel.pkl"


def _safe_import_cleaning_deps():
    # Keep imports lazy so the project can still start even if some NLP deps are missing.
    import contractions

    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from textblob import TextBlob

    return contractions, WordNetLemmatizer, word_tokenize, TextBlob


def _remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def _convert_to_lowercase(text: str) -> str:
    return text.lower()


def _remove_extra_whitespace(text: str) -> str:
    return " ".join(text.split())


def _remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)


def _remove_emails(text: str) -> str:
    return re.sub(r"\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,6}\b", "", text)


def _remove_special_characters(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)


def _remove_specs(text: str) -> str:
    return _remove_special_characters(_remove_emails(_remove_urls(text)))


def _clean_data(text: str) -> str:
    # Mirrors the referenced ISY503 implementation, but with defensive fallbacks.
    # If nltk/textblob resources are missing, we degrade gracefully.
    try:
        contractions, WordNetLemmatizer, word_tokenize, TextBlob = _safe_import_cleaning_deps()

        stop_words = set(ENGLISH_STOP_WORDS)

        try:
            lemmatizer = WordNetLemmatizer()
        except Exception:
            lemmatizer = None

        def _tokenize_words(t: str):
            try:
                return word_tokenize(t)
            except Exception:
                return t.split()

        def remove_stop_words(t: str) -> str:
            words = _tokenize_words(t)
            filtered = [w for w in words if w.lower() not in stop_words]
            return " ".join(filtered)

        def correct_spelling(t: str) -> str:
            try:
                return TextBlob(t).correct().string
            except Exception:
                return t

        def lemmatize_words(t: str) -> str:
            if lemmatizer is None:
                return t

            words = _tokenize_words(t)
            lemmatized = [lemmatizer.lemmatize(w) for w in words]
            return " ".join(lemmatized)

        def expand_contractions(t: str) -> str:
            return contractions.fix(t)

        funcs = [
            _remove_extra_whitespace,
            _remove_punctuation,
            correct_spelling,
            expand_contractions,
            remove_stop_words,
            lemmatize_words,
            _convert_to_lowercase,
            _remove_specs,
        ]

        out = text
        for f in funcs:
            out = f(out)
        return out

    except Exception:
        # Minimal fallback cleaning
        logger.warning("sentiment: clean_data fallback path")
        return _remove_specs(_convert_to_lowercase(_remove_extra_whitespace(text)))


def _word_tokenizer(text: str):
    try:
        _, _, word_tokenize, _ = _safe_import_cleaning_deps()
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    return [w.lower() for w in tokens if w and w not in string.punctuation]


def _to_features(words):
    return {w: True for w in words}


def _load_nb_model():
    global _NB_MODEL
    if _NB_MODEL is not None:
        return _NB_MODEL

    path = _nb_path()
    logger.info("sentiment: loading NB model from %s", path)
    try:
        with open(path, "rb") as f:
            _NB_MODEL = pickle.load(f)
    except Exception:
        logger.exception("sentiment: failed to load NB model from %s", path)
        raise
    return _NB_MODEL


def _load_sk_model():
    global _SK_MODEL
    if _SK_MODEL is not None:
        return _SK_MODEL

    path = _sk_path()
    logger.info("sentiment: loading sklearn model from %s", path)
    try:
        with open(path, "rb") as f:
            _SK_MODEL = pickle.load(f)
    except Exception:
        logger.exception("sentiment: failed to load sklearn model from %s", path)
        raise
    return _SK_MODEL


def _pytorch_preprocess_text(text: str, vocab: dict, max_length: int = 1024):
    try:
        import torch
    except Exception as e:
        raise ModuleNotFoundError(
            "PyTorch sentiment model selected but 'torch' is not installed. Install torch or switch model."
        ) from e

    t0 = time.perf_counter()

    tokens = text.split()
    indices = [vocab.get(tok, vocab.get("<unk>", 0)) for tok in tokens]

    pad_id = vocab.get("<pad>", 0)
    if len(indices) < max_length:
        indices += [pad_id] * (max_length - len(indices))
    else:
        indices = indices[:max_length]

    out = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    dt_ms = int((time.perf_counter() - t0) * 1000)
    logger.warning(
        "sentiment: pytorch_preprocess tokens_n=%s max_length=%s took_ms=%s",
        len(tokens),
        max_length,
        dt_ms,
    )
    return out


def _pytorch_predict(model, text: str, vocab: dict) -> int:
    try:
        import torch
    except Exception as e:
        raise ModuleNotFoundError(
            "PyTorch sentiment model selected but 'torch' is not installed. Install torch or switch model."
        ) from e

    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        input_tensor = _pytorch_preprocess_text(text, vocab)
        t1 = time.perf_counter()
        output = model(input_tensor)
        t2 = time.perf_counter()
        _, predicted = torch.max(output, 1)
        t3 = time.perf_counter()

        logger.warning(
            "sentiment: pytorch_forward took_ms=%s max took_ms=%s total_ms=%s",
            int((t2 - t1) * 1000),
            int((t3 - t2) * 1000),
            int((t3 - t0) * 1000),
        )

        return int(predicted.item())


def _load_pytorch_model_and_vocab():
    global _PT_MODEL, _PT_VOCAB
    if _PT_MODEL is not None and _PT_VOCAB is not None:
        return _PT_MODEL, _PT_VOCAB

    with _PT_LOCK:
        if _PT_MODEL is not None and _PT_VOCAB is not None:
            return _PT_MODEL, _PT_VOCAB

        path = _pt_path()
        logger.warning("sentiment: loading pytorch model+vocab from %s", path)
        try:
            try:
                import dill
            except Exception as e:
                raise ModuleNotFoundError(
                    "PyTorch sentiment model selected but 'dill' is not installed. Install dill or switch model."
                ) from e

            with open(path, "rb") as f:
                data = dill.load(f)
        except Exception:
            logger.exception("sentiment: failed to load pytorch model+vocab from %s", path)
            raise

        _PT_MODEL = data["model"]
        _PT_VOCAB = data["vocab"]
        _PT_MODEL.eval()

        global _PT_WARMED
        if not _PT_WARMED:
            try:
                import torch

                # Limit threads to avoid oversubscription stalls on some setups.
                try:
                    torch.set_num_threads(1)
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass

                t0 = time.perf_counter()
                with torch.no_grad():
                    dummy = torch.zeros((1, 1024), dtype=torch.long)
                    _ = _PT_MODEL(dummy)
                dt_ms = int((time.perf_counter() - t0) * 1000)
                logger.warning("sentiment: pytorch_warmup took_ms=%s", dt_ms)
                _PT_WARMED = True
            except Exception:
                logger.exception("sentiment: pytorch_warmup failed")

        return _PT_MODEL, _PT_VOCAB


def _score_to_result(score: int) -> SentimentResult:
    if score > 0:
        return SentimentResult(score=1, label="positive")
    if score < 0:
        return SentimentResult(score=-1, label="negative")
    return SentimentResult(score=0, label="neutral")


def _normalize_binary_prediction(pred) -> int:
    if pred is None:
        return 0

    if isinstance(pred, bool):
        return 1 if pred else -1

    if isinstance(pred, (int, float)):
        try:
            v = int(pred)
            if v in {-1, 0, 1}:
                return v
            if v in {0, 1, 2}:
                # Common 3-class encoding: 0=negative, 1=neutral, 2=positive
                return {-1: -1, 0: -1, 1: 0, 2: 1}[v]
            return 0
        except Exception:
            return 0

    s = str(pred).strip().lower()
    if s in {"0", "neg", "negative", "-1"}:
        return -1
    if s in {"2", "neu", "neutral"}:
        return 0
    if s in {"1", "pos", "positive", "+1"}:
        return 1

    return 0


def predict_sentiment(text: str, model_id: int) -> SentimentResult:
    # model_id: 1 NB, 2 sklearn stacking, 3 pytorch
    raw_text = str(text or "")
    if not raw_text.strip():
        return _score_to_result(0)

    clean_text = _clean_data(raw_text)
    logger.warning(
        "sentiment: model_id=%s text_len=%s clean_len=%s raw=%r clean=%r",
        model_id,
        len(raw_text),
        len(clean_text),
        _clip(raw_text),
        _clip(clean_text),
    )

    try:
        if int(model_id) == 1:
            model = _load_nb_model()
            tokens = _word_tokenizer(clean_text)
            feats = _to_features(tokens)
            logger.warning(
                "sentiment: nb_input tokens_n=%s tokens_head=%r feats_n=%s",
                len(tokens),
                tokens[:25],
                len(feats),
            )

            pred = model.classify(feats)
            logger.warning("sentiment: raw_pred nb=%r pred_type=%s", pred, type(pred).__name__)
            score = _normalize_binary_prediction(pred)
            res = _score_to_result(score)
            logger.warning("sentiment: normalized nb score=%s label=%s", res.score, res.label)
            return res

        if int(model_id) == 2:
            model = _load_sk_model()
            logger.info("sentiment: sklearn_input clean=%r", _clip(clean_text))
            pred = model.predict([clean_text])
            logger.info("sentiment: raw_pred sklearn=%r pred_type=%s", pred, type(pred).__name__)
            first = pred[0] if hasattr(pred, "__len__") and len(pred) else pred
            score = _normalize_binary_prediction(first)
            res = _score_to_result(score)
            logger.info("sentiment: normalized sklearn score=%s label=%s", res.score, res.label)
            return res

        if int(model_id) == 3:
#            model, vocab = _load_pytorch_model_and_vocab()
#            logger.warning(
#                "sentiment: pytorch_input clean=%r vocab_size=%s",
#                _clip(clean_text),
#                len(vocab) if vocab else 0,
#            )
#            pred = _pytorch_predict(model, clean_text, vocab)
#            logger.warning("sentiment: raw_pred pytorch=%r pred_type=%s", pred, type(pred).__name__)
#            score = _normalize_binary_prediction(pred)
#            res = _score_to_result(score)
#            logger.warning("sentiment: normalized pytorch score=%s label=%s", res.score, res.label)
            return 0

        raise ValueError(f"Unknown sentiment model_id: {model_id}")

    except Exception:
        logger.exception(
            "sentiment: prediction failed model_id=%s raw=%r clean=%r",
            model_id,
            _clip(raw_text),
            _clip(clean_text),
        )
        return _score_to_result(0)
