"""check_prefixes: loan-id prefixes must be pairwise distinct across perf files.

Run from the repo root:  python -m unittest discover tests
"""

import unittest
import warnings

from src.panel import check_prefixes

DISTINCT = {"perf_2006Q1.txt": "F06Q1", "perf_2006Q2.txt": "F06Q2",
            "perf_2007Q1.txt": "F07Q1"}
COLLIDING = {"perf_2006Q1.txt": "F06Q1", "perf_2006Q2.txt": "F06Q1",
             "perf_2007Q1.txt": "F07Q1"}


class CheckPrefixesTest(unittest.TestCase):

    def test_distinct_prefixes_pass(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(check_prefixes(DISTINCT), {})

    def test_collision_raises_naming_prefix_and_files(self):
        with self.assertRaises(SystemExit) as ctx:
            check_prefixes(COLLIDING)
        msg = str(ctx.exception)
        for part in ("F06Q1", "perf_2006Q1.txt", "perf_2006Q2.txt"):
            self.assertIn(part, msg)
        self.assertNotIn("perf_2007Q1.txt", msg)

    def test_override_downgrades_to_warning(self):
        with self.assertWarns(RuntimeWarning) as ctx:
            collisions = check_prefixes(COLLIDING, allow_multi_file_loans=True)
        self.assertEqual(collisions,
                         {"F06Q1": ["perf_2006Q1.txt", "perf_2006Q2.txt"]})
        self.assertIn("F06Q1", str(ctx.warning))


if __name__ == "__main__":
    unittest.main()
