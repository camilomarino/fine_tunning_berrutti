"""Keep standalone scoring equal to the historical ascii-text-v1 policy."""

import unittest

from evaluation.scoring import score_pairs
from evaluation.scoring import normalize_text


class ScoringTests(unittest.TestCase):
    def test_ascii_transliteration_and_whitespace_order(self):
        self.assertEqual(normalize_text("  José\n\tÁlvarez  "), "Jose Alvarez")

    def test_micro_cer_and_wer(self):
        scored = score_pairs([("á bc", "a bd"), ("xyz", "xyz")])
        self.assertEqual(scored["char_errors"], 1)
        self.assertEqual(scored["char_count"], 7)
        self.assertEqual(scored["cer"], 1 / 7)
        self.assertEqual(scored["word_errors"], 1)
        self.assertEqual(scored["word_count"], 3)


if __name__ == "__main__":
    unittest.main()
