"""The HPI join and the loan-level exclusions in src.panel."""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import panel


def write_hpi(path, rows):
    pd.DataFrame(rows, columns=["state_code", "month", "hpi_step", "hpi_disagg"]).to_csv(
        path, index=False)


def series(state, start_year, n, step0, disagg0, growth):
    months = pd.date_range("%d-01-01" % start_year, periods=n, freq="MS")
    return [(state, m.strftime("%Y-%m-%d"), step0 * growth ** (i // 3),
             disagg0 * growth ** (i / 3.0)) for i, m in enumerate(months)]


class HpiTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.path = os.path.join(self.dir, "hpi.csv")

    def month_index(self, y, m):
        return y * 12 + m - 1

    def test_ratio_is_one_at_origination_and_est_ltv_rebases(self):
        write_hpi(self.path, series("CA", 2006, 24, 100.0, 100.0, 1.1)
                  + series("TX", 2006, 24, 100.0, 100.0, 1.0))
        hpi = panel.Hpi(self.path)
        om = np.full(3, self.month_index(2006, 3))
        t = np.array([self.month_index(2006, 3), self.month_index(2006, 12), self.month_index(2007, 6)])
        out = panel.hpi_features(hpi, "2006Q1", "F06Q1", np.arange(3), np.array([0, 0, 1]),
                                 ["CA", "TX"], t, om, np.full(3, 80, dtype=np.float32))
        self.assertEqual(out["hpi_ratio"][0], 1.0)
        self.assertEqual(out["est_ltv"][0], 80.0)
        self.assertAlmostEqual(float(out["hpi_ratio"][1]), 1.1 ** 3, places=5)
        self.assertAlmostEqual(float(out["est_ltv"][1]), 80 / 1.1 ** 3, places=3)
        self.assertEqual(out["hpi_ratio"][2], 1.0)               # TX is flat
        for col in panel.HPI_FEATURES:
            self.assertEqual(out[col].dtype, np.float32)

    def test_step_series_uses_its_own_values(self):
        write_hpi(self.path, series("CA", 2006, 12, 100.0, 100.0, 1.1))
        hpi = panel.Hpi(self.path)
        om, t = np.array([self.month_index(2006, 1)]), np.array([self.month_index(2006, 4)])
        out = panel.hpi_features(hpi, "q", "F", np.arange(1), np.array([0]), ["CA"], t, om,
                                 np.array([80], dtype=np.float32))
        self.assertAlmostEqual(float(out["est_ltv_step"][0]), 80 / 1.1, places=3)

    def test_month_outside_the_series_fails_naming_loans(self):
        write_hpi(self.path, series("CA", 2006, 12, 100.0, 100.0, 1.0))
        hpi = panel.Hpi(self.path)
        with self.assertRaises(panel.HpiError) as cm:
            panel.hpi_features(hpi, "2006Q1", "F06Q1", np.array([7, 8]), np.array([0, 0]), ["CA"],
                               np.array([self.month_index(2006, 5), self.month_index(2007, 5)]),
                               np.array([self.month_index(2006, 1)] * 2),
                               np.full(2, 80, dtype=np.float32))
        self.assertIn("F06Q10000008", str(cm.exception))

    def test_state_without_a_series_fails(self):
        write_hpi(self.path, series("CA", 2006, 12, 100.0, 100.0, 1.0))
        hpi = panel.Hpi(self.path)
        with self.assertRaises(panel.HpiError):
            panel.hpi_features(hpi, "q", "F", np.arange(1), np.array([0]), ["PR"],
                               np.array([self.month_index(2006, 5)]),
                               np.array([self.month_index(2006, 1)]),
                               np.array([80], dtype=np.float32))

    def test_a_gap_in_a_state_series_fails_and_is_not_filled(self):
        rows = series("CA", 2006, 12, 100.0, 100.0, 1.0)
        del rows[5]
        rows += series("TX", 2006, 12, 100.0, 100.0, 1.0)
        write_hpi(self.path, rows)
        with self.assertRaises(panel.HpiError) as cm:
            panel.Hpi(self.path)
        self.assertIn("CA", str(cm.exception))

    def test_duplicate_state_month_fails(self):
        rows = series("CA", 2006, 12, 100.0, 100.0, 1.0)
        write_hpi(self.path, rows + rows[:1])
        with self.assertRaises(panel.HpiError):
            panel.Hpi(self.path)

    def test_missing_file_fails(self):
        with self.assertRaises(panel.HpiError):
            panel.Hpi(os.path.join(self.dir, "absent.csv"))


class ExclusionTest(unittest.TestCase):

    def test_territory_and_null_ltv_tails_are_disjoint(self):
        cats = ["CA", "GU", "PR", "TX", "VI"]
        orig = {"state_categories": cats,
                "tail": np.arange(6),
                "property_state": np.array([0, 1, 2, 3, 4, 3], dtype=np.int8),
                "ltv": np.array([80, 80, np.nan, np.nan, 90, 70], dtype=np.float32)}
        terr, nl = panel.exclusion_tails(orig)
        self.assertEqual(list(terr), [1, 2, 4])
        self.assertEqual(list(nl), [3])             # tail 2 is a territory loan, counted once

    def test_schema_groups(self):
        for col in panel.HPI_FEATURES:
            self.assertIn(col, panel.FEATURES)
            self.assertNotIn(col, panel.METADATA)
            self.assertIn(col, panel.SCHEMA)
        self.assertNotIn("hpi_feature", panel.SCHEMA)
        self.assertEqual(panel.SCHEMA, panel.KEYS + panel.STATE_COLS + panel.FEATURES + panel.METADATA)


if __name__ == "__main__":
    unittest.main()
