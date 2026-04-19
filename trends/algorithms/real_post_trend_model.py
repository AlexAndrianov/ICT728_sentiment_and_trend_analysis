from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


DOMAIN_STOPWORDS = {
    "att",
    "at&t",
    "verizon",
    "facebook",
    "instagram",
    "tiktok",
    "rt",
    "amp",
}

DEFAULT_STOPWORDS = set(ENGLISH_STOP_WORDS).union(DOMAIN_STOPWORDS)


@dataclass
class ModelArtifacts:
    dataframe: pd.DataFrame
    vectorizer: TfidfVectorizer
    reducer: Optional[Any]
    model: MiniBatchKMeans
    feature_matrix: Any
    reduced_matrix: Any


@dataclass
class ClusterMetricSummary:
    n_posts_used: int
    k: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    hashtagged_posts_used: int
    hashtag_top3_hit_rate: float
    cluster_top_hashtags: Dict[int, List[List[Any]]]


def parse_hashtags(raw_value: Any) -> List[str]:
    """Parse comma-separated or JSON-like hashtag strings into a clean list."""
    if raw_value is None:
        return []

    text = str(raw_value).strip()
    if not text:
        return []

    parts = [part.strip() for part in text.split(",") if part.strip()]
    clean_tags: List[str] = []

    for part in parts:
        tag = part.strip().strip("[]").strip().strip('"').strip("'").lstrip("#")
        tag = re.sub(r"\s+", "", tag)
        if not tag:
            continue
        if not re.search(r"[A-Za-z]", tag):
            continue
        clean_tags.append(tag.lower())

    # preserve order, remove duplicates
    return list(dict.fromkeys(clean_tags))


def clean_post_text(text: Any) -> str:
    """Normalize post text for trend modeling."""
    text = "" if text is None else str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_posts_dataframe(json_path: str | Path, min_tokens: int = 3) -> pd.DataFrame:
    """Load the converted dataset and keep only posts with enough text to model."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    df = pd.DataFrame(records)
    if "content" not in df.columns:
        raise ValueError("Expected a 'content' field in the posts JSON file.")

    df = df.copy()
    df["clean_text"] = df["content"].map(clean_post_text)
    df["token_count"] = df["clean_text"].str.split().str.len()
    df["hashtags_list"] = df.get("hashtags", "").map(parse_hashtags)
    df["source_platform"] = df["tweet_id"].astype(str).str.split("_").str[0]

    usable_df = df[df["token_count"] >= min_tokens].copy()
    usable_df.reset_index(drop=True, inplace=True)
    return usable_df


def build_baseline_model(df: pd.DataFrame, n_clusters: int = 10) -> Dict[str, float]:
    """Approximate the existing repo baseline for comparison."""
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df["clean_text"])

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=512,
        n_init=3,
        max_iter=50,
    )
    labels = model.fit_predict(matrix)

    silhouette = silhouette_score(
        matrix,
        labels,
        sample_size=min(400, matrix.shape[0]),
        random_state=42,
    )
    return {
        "n_posts_used": int(len(df)),
        "k": int(n_clusters),
        "silhouette": float(silhouette),
    }


def build_improved_model(
    df: pd.DataFrame,
    k_values: Sequence[int] = (6, 8),
    max_features: int = 2500,
    min_df: int = 3,
    max_df: float = 0.7,
    n_components: int = 40,
    stop_words: Optional[Iterable[str]] = None,
) -> tuple[ModelArtifacts, ClusterMetricSummary]:
    """Train a stronger clustering model on real posts and return metrics."""
    stop_words = list(set(stop_words or DEFAULT_STOPWORDS))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        stop_words=stop_words,
    )
    feature_matrix = vectorizer.fit_transform(df["clean_text"])

    reducer = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_matrix = make_pipeline(reducer, Normalizer(copy=False)).fit_transform(feature_matrix)

    best_labels = None
    best_model = None
    best_summary = None
    best_silhouette = float("-inf")

    for k in k_values:
        model = MiniBatchKMeans(
            n_clusters=int(k),
            random_state=42,
            batch_size=512,
            n_init=3,
            max_iter=50,
        )
        labels = model.fit_predict(reduced_matrix)
        silhouette = silhouette_score(
            reduced_matrix,
            labels,
            sample_size=min(400, len(reduced_matrix)),
            random_state=42,
        )
        db_score = davies_bouldin_score(reduced_matrix[:400], labels[:400])
        ch_score = calinski_harabasz_score(reduced_matrix[:400], labels[:400])

        if silhouette > best_silhouette:
            candidate_df = df.copy()
            candidate_df["cluster"] = labels
            hashtag_summary = evaluate_hashtag_alignment(candidate_df)
            best_silhouette = float(silhouette)
            best_labels = labels
            best_model = model
            best_summary = ClusterMetricSummary(
                n_posts_used=int(len(candidate_df)),
                k=int(k),
                silhouette=float(silhouette),
                davies_bouldin=float(db_score),
                calinski_harabasz=float(ch_score),
                hashtagged_posts_used=int(hashtag_summary["hashtagged_posts_used"]),
                hashtag_top3_hit_rate=float(hashtag_summary["hashtag_top3_hit_rate"]),
                cluster_top_hashtags=hashtag_summary["cluster_top_hashtags"],
            )

    assert best_model is not None and best_labels is not None and best_summary is not None
    final_df = df.copy()
    final_df["cluster"] = best_labels

    artifacts = ModelArtifacts(
        dataframe=final_df,
        vectorizer=vectorizer,
        reducer=reducer,
        model=best_model,
        feature_matrix=feature_matrix,
        reduced_matrix=reduced_matrix,
    )
    return artifacts, best_summary


def get_cluster_top_terms(
    model_artifacts: ModelArtifacts,
    top_n: int = 8,
) -> Dict[int, List[str]]:
    """Extract representative TF-IDF terms for each cluster centroid."""
    feature_names = model_artifacts.vectorizer.get_feature_names_out()
    centers = model_artifacts.model.cluster_centers_

    if model_artifacts.reducer is not None:
        centers = model_artifacts.reducer.inverse_transform(centers)

    output: Dict[int, List[str]] = {}
    for cluster_id, center in enumerate(centers):
        top_indices = center.argsort()[::-1][:top_n]
        terms = [feature_names[index] for index in top_indices]
        output[int(cluster_id)] = terms
    return output


def evaluate_hashtag_alignment(clustered_df: pd.DataFrame) -> Dict[str, Any]:
    """Measure whether clusters align with real hashtags on the posts that have them."""
    tagged_df = clustered_df[clustered_df["hashtags_list"].map(bool)].copy()
    if tagged_df.empty:
        return {
            "hashtagged_posts_used": 0,
            "hashtag_top3_hit_rate": 0.0,
            "cluster_top_hashtags": {},
        }

    cluster_tag_counts: Dict[int, Counter] = defaultdict(Counter)
    for _, row in tagged_df.iterrows():
        cluster_id = int(row["cluster"])
        for hashtag in row["hashtags_list"]:
            cluster_tag_counts[cluster_id][hashtag] += 1

    hits = 0
    cluster_top_hashtags: Dict[int, List[List[Any]]] = {}
    for cluster_id, counter in cluster_tag_counts.items():
        top_tags = counter.most_common(5)
        cluster_top_hashtags[int(cluster_id)] = [[tag, int(count)] for tag, count in top_tags]

    for _, row in tagged_df.iterrows():
        top3_tags = [tag for tag, _ in cluster_tag_counts[int(row["cluster"] )].most_common(3)]
        if any(tag in top3_tags for tag in row["hashtags_list"]):
            hits += 1

    return {
        "hashtagged_posts_used": int(len(tagged_df)),
        "hashtag_top3_hit_rate": float(hits / len(tagged_df)),
        "cluster_top_hashtags": cluster_top_hashtags,
    }


def save_cluster_assignments(df: pd.DataFrame, output_path: str | Path) -> None:
    output_df = df[[
        "tweet_id",
        "source_platform",
        "content",
        "hashtags",
        "hashtags_list",
        "cluster",
    ]].copy()
    output_df.to_json(output_path, orient="records", indent=2, force_ascii=False)


def run_full_pipeline(
    json_path: str | Path,
    assignment_output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    df = load_posts_dataframe(json_path=json_path, min_tokens=3)
    baseline_metrics = build_baseline_model(df)
    artifacts, improved_metrics = build_improved_model(df)
    top_terms = get_cluster_top_terms(artifacts)

    if assignment_output_path is not None:
        save_cluster_assignments(artifacts.dataframe, assignment_output_path)

    return {
        "baseline": baseline_metrics,
        "improved": asdict(improved_metrics),
        "cluster_top_terms": {int(k): v for k, v in top_terms.items()},
    }


if __name__ == "__main__":
    input_path = Path("ml_models_design/posts/converted_social_to_twitter_format.json")
    output_path = Path("ml_models_design/posts/clustered_posts_with_trends.json")

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    results = run_full_pipeline(input_path, assignment_output_path=output_path)
    print(json.dumps(results, indent=2))
