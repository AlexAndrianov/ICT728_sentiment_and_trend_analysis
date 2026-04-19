import unittest
from unittest.mock import patch

from predictor_app.trend_predictor import (
    cluster_by_trends,
    define_cluster_name,
    define_similar_cluster,
)


class _FakeEmbedder:
    def encode(self, texts, **kwargs):
        if isinstance(texts, list) and len(texts) > 1:
            vectors = []
            for text in texts:
                t = str(text).lower()
                if "climate" in t:
                    vectors.append([1.0, 0.0])
                elif "football" in t:
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([0.5, 0.5])
            return vectors

        text = str(texts[0]).lower()
        if "climate" in text:
            return [[1.0, 0.0]]
        if "football" in text:
            return [[0.0, 1.0]]
        return [[0.5, 0.5]]


class _FakeKeyBERT:
    def extract_keywords(self, text, **kwargs):
        lowered = text.lower()
        if "ai" in lowered:
            return [("ai automation", 0.92), ("business workflow", 0.7)]
        return [("football finals", 0.9)]


class TrendPredictorFunctionsTests(unittest.TestCase):
    @patch("sentence_transformers.util.community_detection")
    @patch("predictor_app.trend_predictor._get_embedder")
    def test_cluster_by_trends_returns_tweet_to_cluster_mapping(self, mock_get_embedder, mock_community_detection):
        mock_get_embedder.return_value = _FakeEmbedder()
        mock_community_detection.return_value = [[0, 1], [2]]

        tweets = ["AI improves automation", "AI helps teams", "Football finals today"]
        result = cluster_by_trends(tweets)

        self.assertEqual(result, {0: 0, 1: 0, 2: 1})

    @patch("predictor_app.trend_predictor._get_keybert")
    def test_define_cluster_name_returns_human_readable_name(self, mock_get_keybert):
        mock_get_keybert.return_value = _FakeKeyBERT()

        tweet_to_cluster = {0: 0, 1: 0}
        tweets = ["AI tools automate support tasks", "AI automation boosts productivity"]
        result = define_cluster_name(tweet_to_cluster, tweets)

        self.assertEqual(result[0], "ai automation")

    @patch("predictor_app.trend_predictor._get_embedder")
    def test_define_similar_cluster_reuses_old_name_on_semantic_match(self, mock_get_embedder):
        mock_get_embedder.return_value = _FakeEmbedder()

        new_names = {0: "climate crisis", 1: "football final"}
        old_names = ["Climate Change", "Football News"]
        result = define_similar_cluster(new_names, old_names, similarity_threshold=0.7)

        self.assertEqual(result[0], "Climate Change")
        self.assertEqual(result[1], "Football News")


if __name__ == "__main__":
    unittest.main()
