import unittest

from trends.algorithms.trend_definition import (
    clean_post_text,
    define_trend,
    define_trends_for_clusters,
    match_existing_trend,
)


class TestTrendDefinition(unittest.TestCase):
    def test_clean_post_text_removes_urls_mentions_and_hash(self):
        text = "Check this out @john https://example.com #AI #MachineLearning"
        cleaned = clean_post_text(text)

        self.assertNotIn("@john", cleaned)
        self.assertNotIn("https://example.com", cleaned)
        self.assertIn("ai", cleaned)
        self.assertIn("machinelearning", cleaned)

    def test_define_trend_returns_non_empty_label(self):
        posts = [
            "New AI tools are changing business workflows",
            "AI tools help automate repetitive work",
            "Businesses adopt AI tools for productivity gains",
        ]
        trend = define_trend(posts)

        self.assertIsInstance(trend, str)
        self.assertNotEqual(trend.strip(), "")
        self.assertNotEqual(trend, "Misc Trend")

    def test_define_trend_handles_empty_input(self):
        self.assertEqual(define_trend([]), "Misc Trend")
        self.assertEqual(define_trend(["", "   ", None]), "Misc Trend")

    def test_define_trends_for_clusters_groups_posts(self):
        posts = [
            "Football finals were thrilling this year",
            "The team won the football championship",
            "Climate change is causing more floods",
            "Global warming affects weather patterns",
        ]
        labels = [0, 0, 1, 1]

        trends = define_trends_for_clusters(posts, labels)

        self.assertEqual(set(trends.keys()), {0, 1})
        self.assertIsInstance(trends[0], str)
        self.assertIsInstance(trends[1], str)
        self.assertNotEqual(trends[0], "Misc Trend")
        self.assertNotEqual(trends[1], "Misc Trend")

    def test_define_trends_for_clusters_length_mismatch_raises(self):
        posts = ["one", "two"]
        labels = [0]

        with self.assertRaises(ValueError):
            define_trends_for_clusters(posts, labels)

    def test_match_existing_trend_reuses_similar_name(self):
        trend_name = "Climate Change"
        existing = ["Football Finals", "Climate Change Crisis", "Python Tutorials"]

        matched = match_existing_trend(trend_name, existing, threshold=0.5)

        self.assertEqual(matched, "Climate Change Crisis")

    def test_match_existing_trend_keeps_new_name_when_not_similar(self):
        trend_name = "AI Tools"
        existing = ["Football Finals", "Climate Change"]

        matched = match_existing_trend(trend_name, existing, threshold=0.6)

        self.assertEqual(matched, "AI Tools")


if __name__ == "__main__":
    unittest.main()