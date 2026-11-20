"""
tests/test_lifestat.py
----------------------
Simple checks that the mathematics and data handling work correctly.

Run from the project folder with either:
    python tests/test_lifestat.py
    pytest tests
"""

import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import data_utils as du  # noqa: E402
import report_builder as rb  # noqa: E402
import stats_engine as se  # noqa: E402


def close(a, b, tol=1e-9):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def make_df(rows):
    """rows: list of (sleep, energy) -> a valid LifeStat table."""
    base = pd.date_range("2026-01-01", periods=len(rows))
    return pd.DataFrame(
        {
            "date": base,
            "sleep_hours": [r[0] for r in rows],
            "exercise_minutes": 30.0,
            "water_litres": 2.0,
            "screen_hours": 5.0,
            "study_hours": 4.0,
            "energy": [r[1] for r in rows],
            "mood": "Calm",
            "sleep_quality": "Good",
        }
    )


# ---------------------------------------------------------------- Module 1
def test_joint_probability():
    df = make_df([(8, 8), (7, 5), (6, 8), (7, 7), (5, 3)])
    res = se.joint_probability(df, "sleep_hours", ">=", 7, "energy", ">=", 7)
    assert res["total"] == 5
    assert res["count_a"] == 3      # 8, 7, 7
    assert res["count_b"] == 3      # 8, 8, 7
    assert res["count_both"] == 2   # (8,8) and (7,7)
    assert close(res["probability"], 2 / 5)
    assert res["a_only"] == 1 and res["b_only"] == 1 and res["neither"] == 1


def test_joint_probability_empty_and_zero():
    assert se.joint_probability(du.empty_dataframe(), "sleep_hours", ">=", 7, "energy", ">=", 7) is None
    df = make_df([(5, 3), (6, 4)])
    res = se.joint_probability(df, "sleep_hours", ">=", 7, "energy", ">=", 7)
    assert res["count_both"] == 0 and res["probability"] == 0


# ---------------------------------------------------------------- Module 2
def test_discrete_distribution():
    # Ratings 1, 2, 2, 3  ->  P = 0.25, 0.5, 0.25
    s = pd.Series([1, 2, 2, 3])
    res = se.discrete_distribution(s)
    assert close(res["expected"], 2.0)
    assert close(res["expected_square"], 4.5)
    assert close(res["variance"], 0.5)
    assert close(res["std"], math.sqrt(0.5))
    assert close(res["table"]["Probability"].sum(), 1.0)
    assert res["most_common"] == [2]


def test_discrete_matches_numpy_population_variance():
    s = pd.Series([4, 5, 5, 6, 7, 8, 3, 5, 6, 9])
    res = se.discrete_distribution(s)
    assert close(res["expected"], np.mean(s))
    assert close(res["variance"], np.var(s))  # population variance


# ---------------------------------------------------------------- Module 3
def test_continuous_summary_and_probabilities():
    x = np.array([6.0, 7.0, 7.5, 8.0, 9.0])
    res = se.continuous_summary(pd.Series(x))
    assert close(res["mean"], np.mean(x))
    assert close(res["variance"], np.var(x, ddof=1))
    assert close(res["std"], np.std(x, ddof=1))
    emp = se.empirical_probability_between(x, 7, 9)
    assert emp["count"] == 4 and close(emp["probability"], 0.8)
    assert close(se.normal_probability_between(0, 1, -1, 1), 0.6826894921, 1e-8)
    assert se.normal_probability_between(0, 0, -1, 1) is None


def test_distribution_edge_cases():
    same = np.array([7.0] * 10)
    assert se.density_curve(same) is None
    assert not se.normal_fit_check(same)["reasonable"]
    assert not se.normal_fit_check(np.array([1.0, 2.0, 3.0]))["reasonable"]
    grid, dens = se.density_curve(np.random.default_rng(1).normal(7, 1, 50))
    area = np.trapezoid(dens, grid) if hasattr(np, "trapezoid") else np.trapz(dens, grid)
    assert 0.95 < area < 1.01  # density integrates to about 1
    xs, ys = se.empirical_cdf([3, 1, 2])
    assert list(xs) == [1, 2, 3] and close(ys[-1], 1.0)


# ---------------------------------------------------------------- Module 4
def test_regression_known_values():
    # Worked example: x = 1..5, y = 2,4,5,4,5
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    res = se.regression_analysis(x, y)
    assert res["ok"]
    assert close(res["x_mean"], 3) and close(res["y_mean"], 4)
    assert close(res["b_yx"], 0.6)          # Sxy / Sxx = 6 / 10
    assert close(res["b_xy"], 1.0)          # Sxy / Syy = 6 / 6
    assert close(res["r"], 6 / math.sqrt(60))
    assert close(res["r_squared"], res["b_yx"] * res["b_xy"])  # r^2 = b_yx * b_xy
    assert close(se.predict_y_from_x(res, 3), 4)
    assert close(se.predict_y_from_x(res, 5), 4 + 0.6 * 2)
    assert close(se.predict_x_from_y(res, 5), 3 + 1.0 * 1)
    # compare with SciPy
    lin = stats.linregress(x, y)
    assert close(res["b_yx"], lin.slope) and close(res["a_yx"], lin.intercept)


def test_regression_problem_cases():
    assert not se.regression_analysis([1, 2, 3], [1, 2, 3])["ok"]            # too few
    assert not se.regression_analysis([5] * 8, list(range(8)))["ok"]          # no variation in X
    assert not se.regression_analysis(list(range(8)), [2] * 8)["ok"]          # no variation in Y
    assert not se.regression_analysis([], [])["ok"]                           # empty


# ---------------------------------------------------------------- Module 5
def test_pearson_matches_numpy_and_scipy():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 60)
    y = 0.5 * x + rng.normal(0, 1, 60)
    res = se.pearson_correlation(x, y)
    assert close(res["r"], np.corrcoef(x, y)[0, 1])
    assert close(res["r"], stats.pearsonr(x, y)[0])
    assert -1 <= res["r"] <= 1


def test_pearson_extremes_and_labels():
    x = list(range(10))
    assert close(se.pearson_correlation(x, [2 * v + 1 for v in x])["r"], 1.0)
    assert close(se.pearson_correlation(x, [-3 * v for v in x])["r"], -1.0)
    assert se.describe_correlation(0.05)[0] == "no clear direction"
    assert se.describe_correlation(-0.75) == ("negative", "strong")
    assert se.describe_correlation(0.35) == ("positive", "moderate")
    assert not se.pearson_correlation([1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6])["ok"]


# ---------------------------------------------------------------- Data handling
def test_sample_data_is_valid():
    df = du.load_sample()
    assert 60 <= len(df) <= 90
    assert list(df.columns) == du.ALL_COLUMNS
    assert df["date"].is_unique
    assert df["energy"].between(1, 10).all()


def test_validation_missing_columns_and_bad_rows():
    bad_columns = pd.DataFrame({"date": ["2026-01-01"], "sleep_hours": [7]})
    result = du.validate_dataframe(bad_columns)
    assert not result.ok and "missing" in result.errors[0].lower()

    raw = pd.read_csv(du.SAMPLE_PATH).astype(object)
    raw.loc[0, "sleep_hours"] = 30            # out of range
    raw.loc[1, "energy"] = "abc"              # not a number
    raw.loc[2, "date"] = "not a date"         # bad date
    raw.loc[3, "water_litres"] = None         # missing value
    result = du.validate_dataframe(raw)
    assert result.ok
    assert len(result.data) == len(raw) - 4
    assert len(result.warnings) >= 4


def test_validation_duplicates_and_aliases():
    raw = pd.DataFrame(
        {
            "Date": ["2026-01-01", "2026-01-01", "2026-01-02"],
            "Sleep": [7, 8, 6],
            "Exercise": [30, 40, 0],
            "Water Intake": [2, 2.5, 1.5],
            "Screen Time": [5, 4, 6],
            "Study": [4, 3, 5],
            "Energy Rating": [6, 7.4, 5],
        }
    )
    result = du.validate_dataframe(raw)
    assert result.ok and len(result.data) == 2
    assert any("duplicate" in w.lower() for w in result.warnings)
    assert result.data.loc[0, "sleep_hours"] == 8        # last duplicate kept
    assert result.data.loc[0, "energy"] == 7             # decimals rounded
    assert (result.data["mood"] == du.NOT_RECORDED).all()  # optional columns filled


def test_validation_empty_and_all_invalid():
    assert not du.validate_dataframe(pd.DataFrame()).ok
    raw = pd.read_csv(du.SAMPLE_PATH).head(3)
    raw["energy"] = 99
    assert not du.validate_dataframe(raw).ok


def test_manual_record_rules_and_combining():
    good = {"date": "2026-02-01", "sleep_hours": 7, "exercise_minutes": 20, "water_litres": 2,
            "screen_hours": 4, "study_hours": 3, "energy": 6}
    assert du.row_issues(good) == []
    assert du.row_issues({**good, "energy": 11})
    assert du.row_issues({**good, "sleep_hours": -1})
    assert du.row_issues({**good, "sleep_hours": 20, "study_hours": 8})   # more than 24 h combined
    assert du.row_issues({**good, "date": None})

    base = du.make_record_frame(good)
    newer = du.make_record_frame({**good, "energy": 9})
    combined, replaced = du.combine_records(base, newer)
    assert replaced == 1 and len(combined) == 1 and combined.loc[0, "energy"] == 9
    combined, replaced = du.combine_records(du.empty_dataframe(), base)
    assert replaced == 0 and len(combined) == 1


def test_csv_round_trip():
    df = du.load_sample()
    again = du.validate_dataframe(pd.read_csv(pd.io.common.BytesIO(du.to_csv_bytes(df))))
    assert again.ok and len(again.data) == len(df)
    template = du.validate_dataframe(pd.read_csv(pd.io.common.BytesIO(du.template_csv_bytes())))
    assert template.ok and len(template.data) == 3


# ---------------------------------------------------------------- Reports
def test_reports_handle_all_sizes():
    sample = du.load_sample()
    text = rb.build_report(sample, source="sample")
    assert "LIFESTAT" in text and "Total recorded days : 80" in text
    assert len(rb.summary_csv_table(sample)) > 10

    tiny = sample.head(2)
    assert "Not enough records" in rb.build_report(tiny)
    assert len(rb.summary_csv_table(tiny)) >= 7

    empty = du.empty_dataframe()
    assert "no records yet" in rb.build_report(empty).lower()

    constant = make_df([(7, 6)] * 10)
    assert rb.build_report(constant)  # zero-variance data must not crash
    assert rb.summary_csv_table(constant) is not None


def run_all():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, func in tests:
        func()
        print(f"PASS  {name}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
