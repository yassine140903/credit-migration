"""Loan age and modification (R7), the panel manifest, and the validating loader.

Run from the repo root:  python -m unittest discover tests
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src import panel, panel_io
from src.panel_io import PanelError, load_panel

STAMP = "2026-09-29T12:00:00.000000"
QUARTERS = {"2006Q1": 3, "2006Q2": 2}          # quarter -> rows


Y, P, N = panel.MOD_Y, panel.MOD_P, panel.MOD_NONE


def history(*loans):
    """Loan-sorted arrays from (tail, first_period, [freddie ages], [flags])."""
    tail, period, age, mod = [], [], [], []
    for t, start, ages, flags in loans:
        tail += [t] * len(ages)
        period += list(range(start, start + len(ages)))
        age += ages
        mod += flags
    return (np.array(tail, np.int64), np.array(period, np.int32),
            np.array(age, np.int32), np.array(mod, np.int8))


class ModificationTest(unittest.TestCase):

    def test_age_is_monotonic_for_a_loan_that_resets(self):
        tail, period, age, mod = history((1, 100, [40, 41, 1, 2], [N, N, Y, P]))
        orig = panel.join_origination(
            {"tail": np.array([1]), "orig_month": np.array([60]), "n": 1,
             "fico": np.zeros(1, np.float32), "ltv": np.zeros(1, np.float32),
             "dti": np.zeros(1, np.float32),
             "rate_spread": np.zeros(1, np.float32),
             "property_state": np.zeros(1, np.int8),
             "msa": np.array([None], dtype=object),
             "postal_code": np.array([None], dtype=object),
             "status": {f: np.zeros(1, np.int8)
                        for f in panel.CHECKED + ("postal_code",)}},
            tail)[0]["orig_month"]
        true_age = period - orig
        self.assertEqual(list(true_age), [40, 41, 42, 43])
        self.assertTrue(np.all(np.diff(true_age) > 0))

    def test_is_modified_flips_at_first_flag_and_stays_true(self):
        tail, period, age, mod = history(
            (1, 100, [10, 11, 1, 2, 3, 4], [N, N, Y, P, P, N]),
            (2, 100, [5, 6, 7], [N, N, N]))
        is_mod, months, unflagged = panel.modification_columns(tail, period,
                                                               age, mod)
        self.assertEqual(list(is_mod), [False, False, True, True, True, True,
                                        False, False, False])
        self.assertEqual(list(months[:6]), [-1, -1, 0, 1, 2, 3])
        self.assertFalse(unflagged.any())

    def test_months_since_restarts_at_second_modification(self):
        tail, period, age, mod = history(
            (1, 100, [30, 1, 2, 3, 1, 2], [N, Y, P, P, Y, P]))
        _, months, _ = panel.modification_columns(tail, period, age, mod)
        self.assertEqual(list(months), [-1, 0, 1, 2, 0, 1])

    def test_p_flag_without_y_starts_modification(self):
        tail, period, age, mod = history((1, 100, [3, 4, 5], [N, P, P]))
        is_mod, months, _ = panel.modification_columns(tail, period, age, mod)
        self.assertEqual(list(months), [-1, 0, 1])

    def test_unflagged_reset_is_a_modification(self):
        tail, period, age, mod = history((7, 100, [50, 51, 1, 2], [N, N, N, N]))
        is_mod, months, unflagged = panel.modification_columns(tail, period,
                                                               age, mod)
        self.assertEqual(list(is_mod), [False, False, True, True])
        self.assertEqual(list(months), [-1, -1, 0, 1])
        self.assertEqual(list(np.nonzero(unflagged)[0]), [2])

    def test_reset_on_flagged_loan_without_y_is_not_an_event(self):
        tail, period, age, mod = history((1, 100, [30, 1, 2, 1], [N, Y, P, P]))
        _, months, unflagged = panel.modification_columns(tail, period, age, mod)
        self.assertEqual(list(months), [-1, 0, 1, 2])
        self.assertFalse(unflagged.any())


class OriginationChecksTest(unittest.TestCase):
    """The hard checks on origination values: documented sentinel -> null, valid
    range -> kept, anything else -> OriginationError. One failing case per check."""

    PREFIX = "F06Q1"
    GOOD = {"fico": "700", "first_payment": "200603", "dti": "35", "ltv": "80",
            "orig_rate": "6.5", "property_state": "CA", "msa": "12980",
            "postal_code": "941"}

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.oidx = panel.resolve()[1]

    def load(self, *loans):
        """loans: {tail: {field: raw}} overrides of GOOD -> (orig, tails, status)."""
        lines = []
        for tail, over in loans:
            row = dict(self.GOOD, **over)
            fields = [""] * 31
            for label, (i, _, _) in self.oidx.items():
                fields[i] = row.get(label, "")
            fields[self.oidx["loan"][0]] = "%s%07d" % (self.PREFIX, tail)
            lines.append("|".join(fields))
        path = os.path.join(self.dir, "orig.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(os.linesep.join(lines) + os.linesep)
        orig = panel.load_origination(path, self.oidx, {})
        tails = np.array([t for t, _ in loans], dtype=np.int64)
        feats, _ = panel.join_origination(orig, tails)
        return orig, tails, feats

    def check(self, *loans, reaching=None):
        orig, tails, feats = self.load(*loans)
        use = tails if reaching is None else np.array(reaching, dtype=np.int64)
        f = panel.join_origination(orig, use)[0]
        panel.check_origination("2006Q1", self.PREFIX, orig, use, f["status"])
        return feats

    def assertFails(self, field, raw, *fragments):
        with self.assertRaises(panel.OriginationError) as ctx:
            self.check((1, {}), (2, {field: raw}))
        msg = str(ctx.exception)
        for frag in (field + ":", "1 loans", "F06Q10000002=%r" % raw) + fragments:
            self.assertIn(frag, msg)

    def test_valid_values_pass_and_are_kept(self):
        feats = self.check((1, {}), (2, {"ltv": "6", "dti": "0"}),
                           (3, {"ltv": "105", "dti": "65", "orig_rate": "25"}),
                           (4, {"orig_rate": "0.01", "property_state": "GU"}))
        self.assertEqual(list(feats["ltv"]), [80, 6, 105, 80])
        self.assertEqual(list(feats["dti"]), [35, 0, 65, 35])

    def test_ltv_sentinel_999_is_null(self):
        feats = self.check((1, {"ltv": "999"}))
        self.assertTrue(np.isnan(feats["ltv"][0]))

    def test_dti_sentinel_999_is_null(self):
        feats = self.check((1, {"dti": "999"}))
        self.assertTrue(np.isnan(feats["dti"][0]))

    def test_ltv_outside_the_documented_range_fails(self):
        for raw in ("106", "5", "998", "0", "n/a", ""):
            with self.subTest(raw=raw):
                self.assertFails("ltv", raw)

    def test_dti_outside_the_documented_range_fails(self):
        for raw in ("66", "998", "-1", "n/a", ""):
            with self.subTest(raw=raw):
                self.assertFails("dti", raw)

    def test_interest_rate_outside_zero_to_25_fails(self):
        for raw in ("0", "25.01", "99", ".", "", "-3"):
            with self.subTest(raw=raw):
                self.assertFails("orig_rate", raw)

    def test_property_state_outside_the_whitelist_fails(self):
        for raw in ("XX", "ca", "AS", "99", ""):
            with self.subTest(raw=raw):
                self.assertFails("property_state", raw)

    def test_property_state_whitelist_is_50_states_dc_and_three_territories(self):
        self.assertEqual(len(panel.PROPERTY_STATES), 54)
        for code in ("DC", "GU", "PR", "VI", "CA", "WY"):
            self.assertIn(code, panel.PROPERTY_STATES)
        self.assertEqual(panel.PROPERTY_STATES, sorted(panel.PROPERTY_STATES))

    def test_postal_code_000_is_null_and_others_keep_their_zeros(self):
        feats = self.check((1, {"postal_code": "000"}), (2, {"postal_code": "010"}))
        self.assertIsNone(feats["postal_code"][0])
        self.assertEqual(feats["postal_code"][1], "010")
        self.assertEqual(list(feats["status"]["postal_code"]),
                         [panel.SENTINEL, panel.VALID])

    def test_a_malformed_loan_that_never_reaches_the_panel_does_not_fail(self):
        # the F07Q40059296 case: rate 0 in the file, no panel row
        self.check((1, {}), (9, {"orig_rate": "0"}), reaching=[1])
        with self.assertRaises(panel.OriginationError):
            self.check((1, {}), (9, {"orig_rate": "0"}), reaching=[1, 9])

    def test_failure_reports_counts_and_example_loan_ids(self):
        with self.assertRaises(panel.OriginationError) as ctx:
            self.check(*[(t, {"ltv": "500"}) for t in range(1, 9)])
        msg = str(ctx.exception)
        self.assertIn("8 loans (8 panel rows)", msg)
        self.assertEqual(msg.count("F06Q1000000"), panel.EXAMPLES)

    def test_explicit_exclusion_list_names_its_loan_and_reason(self):
        self.assertEqual(list(panel.excluded_tails("F06Q3")), [148547])
        self.assertEqual(len(panel.excluded_tails("F06Q1")), 0)
        self.assertTrue(panel.EXCLUDED_LOANS["F06Q30148547"])


def make_frame(quarter, n):
    """A tiny frame with the panel's real column set and dtypes."""
    codes = np.arange(n, dtype=np.int8) % 3
    return pd.DataFrame({
        "loan_id": pd.array(["F%s%07d" % (quarter[2:], i) for i in range(n)],
                            dtype="string"),
        "from_period": np.full(n, 200601, dtype=np.int32),
        "from_state": pd.Categorical.from_codes(np.zeros(n, np.int8),
                                                  categories=panel.STATES),
        "to_state": pd.Categorical.from_codes(np.zeros(n, np.int8),
                                                categories=panel.STATES),
        "run_length": np.ones(n, dtype=np.int32),
        "run_length_censored": np.zeros(n, dtype=bool),
        "loan_age": pd.array(np.arange(n, dtype=np.int32), dtype="Int32"),
        "is_modified": np.arange(n) >= 1,
        "months_since_modification": pd.array(
            [None] + list(range(n - 1)), dtype="Int32"),
        "fico": np.full(n, 700, dtype=np.float32),
        "ltv": np.full(n, 80, dtype=np.float32),
        "dti": np.full(n, 35, dtype=np.float32),
        "rate_spread": np.zeros(n, dtype=np.float32),
        "unemp_level": np.full(n, 5, dtype=np.float32),
        "unemp_chg_12m": np.zeros(n, dtype=np.float32),
        "hpi_ratio": np.ones(n, dtype=np.float32),
        "est_ltv": np.full(n, 80, dtype=np.float32),
        "est_ltv_step": np.full(n, 80, dtype=np.float32),
        "property_state": pd.Categorical.from_codes(
            np.where(np.arange(n) == 1, -1, 0).astype(np.int8),
            categories=["CA", "TX"]),
        "role": pd.Categorical.from_codes(codes, categories=panel.ROLES),
        "post_gap": np.zeros(n, dtype=bool),
        "is_skip": np.zeros(n, dtype=bool),
        "is_fall": np.zeros(n, dtype=bool),
        "msa": pd.array(["12980"] + [None] * (n - 1), dtype="string"),
        "postal_code": pd.array(["010"] * n, dtype="string"),
    })[panel.SCHEMA]


class LoaderTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        patcher = mock.patch.object(panel_io, "PANEL_DIR", self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        rep = panel.Report()
        rep.build_stamp = STAMP
        for q, n in QUARTERS.items():
            rep.schemas[q] = panel.write_partition(make_frame(q, n),
                                                   self.path(q), STAMP)
            rep.quarters.append((q, n, 0.0))
        rep.quarters.append(("2006Q3", 0, 0.0))   # zero rows: no file written
        self.manifest = panel.write_manifest(rep, self.dir)

    def path(self, quarter):
        return os.path.join(self.dir, panel.PANEL_FILE % quarter)

    def assertPanelError(self, *fragments):
        with self.assertRaises(PanelError) as ctx:
            load_panel()
        for frag in fragments:
            self.assertIn(frag, str(ctx.exception))

    def test_manifest_counts_only_written_partitions(self):
        self.assertEqual(self.manifest["n_partitions"], 2)
        self.assertEqual(self.manifest["total_rows"], 5)

    def test_valid_panel_loads(self):
        df = load_panel()
        self.assertEqual(len(df), 5)
        self.assertEqual(list(df.columns), panel.SCHEMA)
        self.assertEqual(str(df["loan_age"].dtype), "Int32")
        self.assertIsInstance(df["role"].dtype, pd.CategoricalDtype)
        self.assertEqual(str(df["is_modified"].dtype), "bool")
        self.assertEqual(str(df["months_since_modification"].dtype), "Int32")

    def test_manifest_records_modification_columns(self):
        types = {c["name"]: c["type"] for c in self.manifest["schema"]}
        self.assertEqual(types["is_modified"], "bool")
        self.assertEqual(types["months_since_modification"], "int32")
        self.assertEqual([c["name"] for c in self.manifest["schema"]],
                         panel.SCHEMA)

    def test_property_state_is_a_string_dictionary_feature(self):
        types = {c["name"]: c["type"] for c in self.manifest["schema"]}
        self.assertEqual(types["property_state"],
                         "dictionary<values=string, indices=int8, ordered=0>")
        self.assertIn("property_state", panel.FEATURES)
        self.assertNotIn("property_state", panel.METADATA)
        for col in ("msa", "postal_code"):
            self.assertIn(col, panel.METADATA)
            self.assertNotIn(col, panel.FEATURES)
        df = load_panel()
        self.assertIsInstance(df["property_state"].dtype, pd.CategoricalDtype)
        self.assertEqual(list(df["property_state"].isna()),
                         [False, True, False, False, True])   # row 1 of each file
        self.assertEqual(set(df["property_state"].dropna()), {"CA"})
        self.assertEqual(df["postal_code"].iloc[0], "010")    # leading zero kept

    def test_property_state_with_another_type_raises(self):
        # a numeric property_state - in the files AND recorded in the manifest -
        # must still be refused: the declared type is not the manifest's to set
        bad = {q: make_frame(q, n).assign(
            property_state=np.zeros(n, dtype=np.float32))
            for q, n in QUARTERS.items()}
        rep = panel.Report()
        rep.build_stamp = STAMP
        for q, n in QUARTERS.items():
            rep.schemas[q] = panel.write_partition(bad[q], self.path(q), STAMP)
            rep.quarters.append((q, n, 0.0))
        panel.write_manifest(rep, self.dir)
        self.assertPanelError("column property_state", "float",
                              "the schema declares")

    def test_missing_partition_raises(self):
        os.remove(self.path("2006Q2"))
        self.assertPanelError("panel_2006Q2.parquet",
                              "listed in the manifest, missing on disk")

    def test_row_count_mismatch_raises(self):
        panel.write_partition(make_frame("2006Q1", 4), self.path("2006Q1"),
                              STAMP)
        self.assertPanelError("panel_2006Q1.parquet", "4 rows on disk",
                              "manifest says 3")

    def test_missing_manifest_raises(self):
        os.remove(os.path.join(self.dir, panel.MANIFEST_NAME))
        self.assertPanelError("must be rebuilt")

    def test_different_build_stamp_raises(self):
        panel.write_partition(make_frame("2006Q2", 2), self.path("2006Q2"),
                              "2025-01-01T00:00:00.000000")
        self.assertPanelError("panel_2006Q2.parquet", "build stamp",
                              "2025-01-01T00:00:00.000000", STAMP)

    def test_column_subset_returns_only_those_columns(self):
        df = load_panel(columns=["run_length", "loan_id"])
        self.assertEqual(list(df.columns), ["run_length", "loan_id"])
        self.assertEqual(len(df), 5)

    def test_role_and_quarter_filters(self):
        df = load_panel(roles=["fit"], columns=["loan_id"], quarters=["2006Q1"])
        self.assertEqual(list(df.columns), ["loan_id"])
        self.assertEqual(list(df["loan_id"]), ["F06Q10000000"])


if __name__ == "__main__":
    unittest.main()
