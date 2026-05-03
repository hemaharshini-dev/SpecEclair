import unittest
from src.ingest import normalize_id, extract_title

class TestIngestUtils(unittest.TestCase):
    def test_normalize_id(self):
        self.assertEqual(normalize_id("IS 269 : 1989"), "IS 269: 1989")
        self.assertEqual(normalize_id("IS 1489(PART 1):1991"), "IS 1489 (Part 1): 1991")
        self.assertEqual(normalize_id("IS 455-1989"), "IS 455: 1989") # wait, does it handle -?
        
    def test_extract_title(self):
        self.assertEqual(extract_title("IS 269: 1989 ORDINARY PORTLAND CEMENT"), "ORDINARY PORTLAND CEMENT")
        self.assertEqual(extract_title("IS 455: 1989 PORTLAND SLAG CEMENT (fourth revision)"), "PORTLAND SLAG CEMENT")

if __name__ == "__main__":
    unittest.main()
