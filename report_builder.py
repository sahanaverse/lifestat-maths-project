"""
report_builder.py
-----------------
Creates the plain-text report and the CSV summary that users can download.
The report uses everyday language so it is easy to read and share.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

import data_utils as du
import stats_engine as se

SECTION_AVERAGES = "Average lifestyle measurements"
SECTION_JOINT = "Sleep & energy insights"
SECTION_ENERGY = "Energy patterns"
SECTION_SLEEP = "Sleep duration insights"
SECTION_REGRESSION = "Lifestyle relationships (sleep and energy)"
SECTION_CORRELATION = "Habit connections"
SECTION_OBSERVATIONS = "Important observations"

ALL_SECTIONS = [
    SECTION_AVERAGES,
    SECTION_JOINT,
    SECTION_ENERGY,
    SECTION_SLEEP,
    SECTION_REGRESSION,
    SECTION_CORRELATION,
    SECTION_OBSERVATIONS,
]

SOURCE_TEXT = {
    "sample": "Built-in sample data (illustrative, not real records)",
    "mixed": "A mix of built-in sample data and user records",
    "user": "User-entered or uploaded records",
}


def averages_table(df: pd.DataFrame) -> pd.DataFrame:
    """Average of every numeric lifestyle measurement."""
    rows = []
    for col in du.NUMERIC_COLUMNS:
        rows.append({"Measurement": du.LABELS[col], "Average": round(float(df[col].mean()), 2)})
    return pd.DataFrame(rows)


def strongest_connection(df: pd.DataFrame):
    """Find the pair of numeric variables with the strongest Pearson correlation."""
    best = None
    for a, b in combinations(du.NUMERIC_COLUMNS, 2):
        result = se.pearson_correlation(df[a], df[b], a, b)
        if result["ok"] and (best is None or abs(result["r"]) > abs(best[2]["r"])):
            best = (a, b, result)
    return best


def summary_csv_table(df: pd.DataFrame) -> pd.DataFrame:
    """A tidy table of key results for the CSV download."""
    rows = [{"Section": "Overview", "Measure": "Total recorded days", "Value": len(df)}]
    for col in du.NUMERIC_COLUMNS:
        rows.append({"Section": "Averages", "Measure": du.LABELS[col], "Value": round(float(df[col].mean()), 3)})

    joint = se.joint_probability(df, "sleep_hours", ">=", 7.0, "energy", ">=", 7.0)
    if joint:
        rows.append({"Section": "Sleep & energy", "Measure": "Days with sleep >= 7 h AND energy >= 7",
                     "Value": joint["count_both"]})
        rows.append({"Section": "Sleep & energy", "Measure": "Probability of both", "Value": round(joint["probability"], 4)})

    dist = se.discrete_distribution(df["energy"])
    if dist:
        rows.append({"Section": "Energy patterns", "Measure": "Expected energy rating", "Value": round(dist["expected"], 3)})
        rows.append({"Section": "Energy patterns", "Measure": "Variance", "Value": round(dist["variance"], 3)})
        rows.append({"Section": "Energy patterns", "Measure": "Standard deviation", "Value": round(dist["std"], 3)})

    sleep = se.continuous_summary(df["sleep_hours"])
    if sleep and sleep["n"] > 1:
        rows.append({"Section": "Sleep duration", "Measure": "Variance (hours squared)", "Value": round(sleep["variance"], 3)})
        rows.append({"Section": "Sleep duration", "Measure": "Standard deviation (hours)", "Value": round(sleep["std"], 3)})

    reg = se.regression_analysis(df["sleep_hours"], df["energy"], "Sleep", "Energy")
    if reg["ok"]:
        rows.append({"Section": "Relationships", "Measure": "Energy on Sleep: coefficient b_yx", "Value": round(reg["b_yx"], 4)})
        rows.append({"Section": "Relationships", "Measure": "Sleep on Energy: coefficient b_xy", "Value": round(reg["b_xy"], 4)})
        rows.append({"Section": "Relationships", "Measure": "R-squared", "Value": round(reg["r_squared"], 4)})
        rows.append({"Section": "Habit connections", "Measure": "Pearson r (sleep, energy)", "Value": round(reg["r"], 4)})
    return pd.DataFrame(rows)


def build_report(df: pd.DataFrame, sections=None, source="user") -> str:
    """Build the full text report for the chosen sections."""
    sections = list(sections) if sections else list(ALL_SECTIONS)
    lines = []
    add = lines.append

    add("=" * 60)
    add("LIFESTAT - PERSONAL LIFESTYLE REPORT")
    add("=" * 60)
    add(f"Data source : {SOURCE_TEXT.get(source, SOURCE_TEXT['user'])}")
    add(f"Total recorded days : {len(df)}")
    if len(df) > 0:
        first = pd.to_datetime(df["date"]).min().strftime("%d %b %Y")
        last = pd.to_datetime(df["date"]).max().strftime("%d %b %Y")
        add(f"Period covered : {first} to {last}")
    add("")

    if len(df) == 0:
        add("There are no records yet. Add records or upload a CSV file to see results.")
        return "\n".join(lines)

    # ---- Averages ---------------------------------------------------------
    if SECTION_AVERAGES in sections:
        add("-" * 60)
        add(SECTION_AVERAGES.upper())
        add("-" * 60)
        for _, row in averages_table(df).iterrows():
            add(f"  {row['Measurement']:<32} {row['Average']:.2f}")
        add("")

    # ---- Module 1 ---------------------------------------------------------
    if SECTION_JOINT in sections:
        add("-" * 60)
        add(SECTION_JOINT.upper())
        add("-" * 60)
        joint = se.joint_probability(df, "sleep_hours", ">=", 7.0, "energy", ">=", 7.0)
        add("  Events: sleeping at least 7 hours AND an energy rating of at least 7.")
        add(f"  Days with at least 7 hours of sleep : {joint['count_a']}")
        add(f"  Days with energy rating 7 or more   : {joint['count_b']}")
        add(f"  Days with both together             : {joint['count_both']} of {joint['total']}")
        add(f"  Chance of both on a recorded day    : {joint['probability']:.1%}")
        add("")

    # ---- Module 2 ---------------------------------------------------------
    if SECTION_ENERGY in sections:
        add("-" * 60)
        add(SECTION_ENERGY.upper())
        add("-" * 60)
        if len(df) < se.MIN_RECORDS:
            add(f"  Not enough records yet (at least {se.MIN_RECORDS} needed).")
        else:
            dist = se.discrete_distribution(df["energy"])
            common = ", ".join(str(v) for v in dist["most_common"])
            add(f"  Expected (average) energy rating : {dist['expected']:.2f}")
            add(f"  Variance                         : {dist['variance']:.2f}")
            add(f"  Standard deviation               : {dist['std']:.2f}")
            add(f"  Most common rating(s)            : {common}")
            add(f"  Share of days rated 7 or higher  : {dist['p_high']:.1%}")
        add("")

    # ---- Module 3 ---------------------------------------------------------
    if SECTION_SLEEP in sections:
        add("-" * 60)
        add(SECTION_SLEEP.upper())
        add("-" * 60)
        if len(df) < se.MIN_RECORDS:
            add(f"  Not enough records yet (at least {se.MIN_RECORDS} needed).")
        else:
            summary = se.continuous_summary(df["sleep_hours"])
            share = se.empirical_probability_between(summary["values"], 7.0, 9.0)
            add(f"  Average sleep                    : {summary['mean']:.2f} hours")
            add(f"  Variance                         : {summary['variance']:.2f}")
            add(f"  Standard deviation               : {summary['std']:.2f} hours")
            add(f"  Days with 7 to 9 hours of sleep  : {share['count']} of {share['total']} ({share['probability']:.1%})")
            add("  (This share is counted directly from your records; no bell-curve assumption.)")
        add("")

    # ---- Module 4 ---------------------------------------------------------
    if SECTION_REGRESSION in sections:
        add("-" * 60)
        add(SECTION_REGRESSION.upper())
        add("-" * 60)
        reg = se.regression_analysis(df["sleep_hours"], df["energy"], "Sleep", "Energy")
        if not reg["ok"]:
            add("  " + reg["message"])
        else:
            add(f"  Energy = {reg['a_yx']:.2f} + {reg['b_yx']:.2f} x Sleep")
            add(f"  Sleep  = {reg['a_xy']:.2f} + {reg['b_xy']:.2f} x Energy")
            add(f"  R-squared: {reg['r_squared']:.2f} (share of the ups and downs in energy")
            add("  that a straight-line link with sleep describes).")
            add("  Note: this shows an estimated relationship, not a proven cause.")
        add("")

    # ---- Module 5 ---------------------------------------------------------
    if SECTION_CORRELATION in sections:
        add("-" * 60)
        add(SECTION_CORRELATION.upper())
        add("-" * 60)
        corr = se.pearson_correlation(df["sleep_hours"], df["energy"], "Sleep", "Energy")
        if not corr["ok"]:
            add("  " + corr["message"])
        else:
            add(f"  Pearson correlation (sleep and energy): {corr['r']:.2f}")
            add(f"  This is a {corr['strength']} {corr['direction']} linear association.")
            add("  Correlation does not prove that one habit causes the other.")
        add("")

    # ---- Observations -----------------------------------------------------
    if SECTION_OBSERVATIONS in sections:
        add("-" * 60)
        add(SECTION_OBSERVATIONS.upper())
        add("-" * 60)
        for text in observations(df):
            add(f"  * {text}")
        add("")

    add("-" * 60)
    add("NOTES")
    add("  * Results describe only the records provided and can change as data grows.")
    add("  * Small datasets give less reliable results.")
    add("  * LifeStat is an educational statistics project, not medical advice.")
    return "\n".join(lines)


def observations(df: pd.DataFrame) -> list:
    """Short plain-language observations drawn from the data."""
    notes = []
    if len(df) == 0:
        return ["No records yet."]

    avg_sleep = float(df["sleep_hours"].mean())
    notes.append(f"Average sleep is {avg_sleep:.1f} hours per night across {len(df)} recorded days.")

    nights_7 = float((df["sleep_hours"] >= 7).mean())
    notes.append(f"You slept at least 7 hours on {nights_7:.0%} of recorded days.")

    if len(df) >= se.MIN_RECORDS:
        dist = se.discrete_distribution(df["energy"])
        notes.append(f"The typical (expected) energy rating is {dist['expected']:.1f} out of 10.")

        best = strongest_connection(df)
        if best:
            a, b, result = best
            notes.append(
                f"The strongest connection found is between {du.LABELS[a]} and {du.LABELS[b]}: "
                f"a {result['strength']} {result['direction']} association (r = {result['r']:.2f})."
            )
    else:
        notes.append(f"Add at least {se.MIN_RECORDS} records to unlock distribution and relationship results.")
    return notes
