import unittest

import pandas as pd

from trends.algorithms.real_post_trend_model import (
    clean_post_text,
    evaluate_hashtag_alignment,
    parse_hashtags,
)


class TestRealPostTrendModel(unittest.TestCase):
    def test_parse_hashtags_handles_mixed_strings(self):
        raw = 'crm,solopreneur,["#crm","#solopreneur"]'
        parsed = parse_hashtags(raw)
        self.assertEqual(parsed, ["crm", "solopreneur"])

    def test_parse_hashtags_filters_numeric_only_entries(self):
        raw = '3,8,#AI'
        parsed = parse_hashtags(raw)
        self.assertEqual(parsed, ["ai"])

    def test_clean_post_text_removes_urls_mentions_and_hash_symbol(self):
        text = "Check @alex https://example.com #AI tools now"
        cleaned = clean_post_text(text)
        self.assertEqual(cleaned, "check ai tools now")

    def test_evaluate_hashtag_alignment_returns_expected_hit_rate(self):
        df = pd.DataFrame(
            [
                {"cluster": 0, "hashtags_list": ["crm", "sales"]},
                {"cluster": 0, "hashtags_list": ["crm"]},
                {"cluster": 1, "hashtags_list": ["sports"]},
                {"cluster": 1, "hashtags_list": ["sports", "final"]},
            ]
        )
        result = evaluate_hashtag_alignment(df)
        self.assertEqual(result["hashtagged_posts_used"], 4)
        self.assertAlmostEqual(result["hashtag_top3_hit_rate"], 1.0)
        self.assertIn(0, result["cluster_top_hashtags"])

    def test_evaluate_hashtag_alignment_handles_no_hashtags(self):
        df = pd.DataFrame([
            {"cluster": 0, "hashtags_list": []},
            {"cluster": 1, "hashtags_list": []},
        ])
        result = evaluate_hashtag_alignment(df)
        self.assertEqual(result["hashtagged_posts_used"], 0)
        self.assertEqual(result["hashtag_top3_hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
