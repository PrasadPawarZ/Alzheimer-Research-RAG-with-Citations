import unittest

from text_utils import (
    chunk_text,
    count_tokens,
    keyword_overlap_score,
    validate_citation_numbers,
)


class TextUtilsTests(unittest.TestCase):
    def test_chunk_text_respects_token_budget_approximately(self):
        text = (
            "CNN models classify MRI scans. They report accuracy and sensitivity. "
            "Feature extraction improves diagnosis. Explainable AI helps clinicians."
        )
        chunks = chunk_text(text, chunk_size_tokens=10, overlap_tokens=3)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(count_tokens(chunk) <= 16 for chunk in chunks))

    def test_keyword_overlap_score_rewards_matching_terms(self):
        query = "CNN accuracy MRI classification"
        good = "The CNN model reports MRI classification accuracy."
        poor = "The paper discusses hospital scheduling."
        self.assertGreater(keyword_overlap_score(query, good), keyword_overlap_score(query, poor))

    def test_citation_validation(self):
        self.assertTrue(validate_citation_numbers("CNN accuracy is high [1].", 2))
        self.assertFalse(validate_citation_numbers("CNN accuracy is high.", 2))
        self.assertFalse(validate_citation_numbers("CNN accuracy is high [3].", 2))


if __name__ == "__main__":
    unittest.main()
