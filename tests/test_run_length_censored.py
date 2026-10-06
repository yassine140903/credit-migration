"""run_length_censored: the artificial-start flag, over each loan's full history."""

import unittest

import numpy as np

from src import panel

M0 = 24000            # an arbitrary contiguous month index


def build(loans, orig_months):
    """loans: list of (tail, [(month, state)]), rows already in (loan, month) order.
    Mirrors build_quarter's step 3, then calls the function under test."""
    tail = np.array([t for t, rows in loans for _ in rows], dtype=np.int64)
    period = np.array([m for t, rows in loans for m, _ in rows], dtype=np.int32)
    state = np.array([s for t, rows in loans for _, s in rows])
    n = len(tail)
    is_first = np.ones(n, dtype=bool)
    is_first[1:] = tail[1:] != tail[:-1]
    dp = np.zeros(n, dtype=np.int32)
    dp[1:] = period[1:] - period[:-1]
    ar = np.arange(n)
    new_run = np.empty(n, dtype=bool)
    new_run[0] = True
    new_run[1:] = is_first[1:] | (dp[1:] != 1) | (state[1:] != state[:-1])
    run_start = np.maximum.accumulate(np.where(new_run, ar, 0))
    gap_start = np.zeros(n, dtype=bool)
    gap_start[1:] = ~is_first[1:] & (dp[1:] != 1)
    keys = sorted(orig_months)
    orig = {"n": len(keys), "tail": np.array(keys, dtype=np.int64),
            "orig_month": np.array([orig_months[k] for k in keys], dtype=np.int64)}
    return panel_call(tail, period, dp, is_first, new_run, run_start, gap_start, orig)


def panel_call(tail, period, dp, is_first, new_run, run_start, gap_start, orig):
    return panel.run_length_censored("2006Q1", "F06Q1", tail, period, dp, is_first, new_run,
                                     run_start, gap_start, orig)


C, D = "C", "D"


class FlagTest(unittest.TestCase):

    def flags(self, loans, orig):
        censored, kind, entry = build(loans, orig)
        out, i = {}, 0
        for t, rows in loans:
            out[t] = (list(censored[i:i + len(rows)]), list(kind[i:i + len(rows)]))
            i += len(rows)
        return out, list(entry)

    def test_loan_entering_at_origination_is_never_censored(self):
        rows = [(M0 + 1 + k, s) for k, s in enumerate([C, C, D, C, C, C])]   # first row: loan_age 1
        out, entry = self.flags([(1, rows)], {1: M0})
        self.assertEqual(out[1][0], [False] * 6)
        self.assertEqual(entry, [])

    def test_loan_age_zero_at_entry_is_not_censored_either(self):
        out, _ = self.flags([(1, [(M0, C), (M0 + 1, C)])], {1: M0})
        self.assertEqual(out[1][0], [False, False])

    def test_seasoned_entrant_is_flagged_until_the_first_state_change(self):
        rows = [(M0 + 5 + k, s) for k, s in enumerate([C, C, C, D, D])]      # first row: loan_age 5
        out, entry = self.flags([(1, rows)], {1: M0})
        self.assertEqual(out[1][0], [True, True, True, False, False])
        self.assertEqual(out[1][1], [1, 1, 1, 0, 0])
        self.assertEqual(entry, [1])

    def test_row_after_a_gap_starts_a_censored_run_that_ends_at_a_state_change(self):
        rows = [(M0 + 1, C), (M0 + 2, C), (M0 + 5, C), (M0 + 6, C), (M0 + 7, D), (M0 + 8, D)]
        out, entry = self.flags([(1, rows)], {1: M0})
        self.assertEqual(out[1][0], [False, False, True, True, False, False])
        self.assertEqual(out[1][1], [0, 0, 2, 2, 0, 0])
        self.assertEqual(entry, [])

    def test_a_state_change_across_the_gap_does_not_count_as_observed(self):
        rows = [(M0 + 1, C), (M0 + 4, D), (M0 + 5, D), (M0 + 6, C)]
        out, _ = self.flags([(1, rows)], {1: M0})
        self.assertEqual(out[1][0], [False, True, True, False])

    def test_entry_censoring_runs_through_a_later_gap(self):
        rows = [(M0 + 6, C), (M0 + 7, C), (M0 + 10, C), (M0 + 11, D)]
        out, _ = self.flags([(1, rows)], {1: M0})
        self.assertEqual(out[1][0], [True, True, True, False])
        self.assertEqual(out[1][1], [1, 1, 2, 0])      # entry cause, then gap cause after the hole

    def test_loans_do_not_leak_into_each_other(self):
        a = [(M0 + 6, C), (M0 + 7, C)]
        b = [(M0 + 1, C), (M0 + 2, C)]
        out, entry = self.flags([(1, a), (2, b)], {1: M0, 2: M0})
        self.assertEqual(out[1][0], [True, True])
        self.assertEqual(out[2][0], [False, False])
        self.assertEqual(entry, [1])

    def test_loan_without_an_origination_record_is_not_flagged_at_entry(self):
        out, entry = self.flags([(9, [(M0 + 50, C), (M0 + 51, C)])], {1: M0})
        self.assertEqual(out[9][0], [False, False])
        self.assertEqual(entry, [])

    def test_a_broken_invariant_is_a_hard_failure(self):
        # an artificial start in the middle of an unbroken run: the flag would go false then true
        tail = np.array([1, 1, 1], dtype=np.int64)
        period = np.array([M0 + 6, M0 + 7, M0 + 8], dtype=np.int32)      # consecutive months
        dp = np.array([0, 1, 1], dtype=np.int32)
        is_first = np.array([True, False, False])
        new_run = np.array([True, False, True])
        run_start = np.array([0, 0, 2])
        gap_start = np.array([False, False, True])      # claims a gap that dp says is not there
        orig = {"n": 1, "tail": np.array([1]), "orig_month": np.array([M0 + 5])}    # entry at loan_age 1
        with self.assertRaises(RuntimeError) as cm:
            panel_call(tail, period, dp, is_first, new_run, run_start, gap_start, orig)
        self.assertIn("false then true", str(cm.exception))

    def test_schema_group(self):
        self.assertIn("run_length_censored", panel.FEATURES)
        self.assertNotIn("run_length_censored", panel.METADATA)
        self.assertIn("run_length_censored", panel.SCHEMA)
        self.assertIn("post_gap", panel.METADATA)         # unchanged


if __name__ == "__main__":
    unittest.main()
