import unittest

from translate import detect_language, to_english


class TranslateTests(unittest.TestCase):
    def test_detects_devanagari_hindi(self):
        self.assertEqual(detect_language("\u0905\u0932\u094d\u091c\u093c\u093e\u0907\u092e\u0930 \u0915\u094d\u092f\u093e \u0939\u0948?"), "hi")

    def test_detects_romanized_hindi(self):
        self.assertEqual(detect_language("alzaimer kya hai?"), "hi")
        self.assertEqual(detect_language("alzheimer kya hota hai?"), "hi")
        self.assertEqual(detect_language("mri kya hai?"), "hi")

    def test_keeps_plain_english_as_english(self):
        self.assertEqual(detect_language("what is alzheimer?"), "en")

    def test_normalizes_romanized_hindi_query_without_llm(self):
        self.assertEqual(to_english("alzaimer kya hai?", "hi"), "what is alzheimer?")
        self.assertEqual(to_english("mri kya hai?", "hi"), "what is mri?")


if __name__ == "__main__":
    unittest.main()
