"""
stats_engine.py
---------------
All the mathematics behind LifeStat. Nothing here uses Streamlit, so every
function can be tested on its own (see tests/test_lifestat.py).

Modules
  1. Joint probability            -> joint_probability()
  2. Discrete distribution        -> discrete_distribution()
  3. Continuous distribution      -> continuous_summary(), density_curve(), ...
  4. Regression lines             -> regression_analysis()
  5. Pearson correlation          -> pearson_correlation()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MIN_RECORDS = 5  # fewest records needed for the distribution/relationship modules

# Conditions a user can choose for an event.
OPERATORS = {
    ">=": lambda s, v: s >= v,
    "<=": lambda s, v: s <= v,
    ">": lambda s, v: s > v,
    "<": lambda s, v: s < v,
}


# ---------------------------------------------------------------------------
# MODULE 1 - Joint probability:  P(A and B) = (days with both) / (all days)
# ---------------------------------------------------------------------------
def joint_probability(df, col_a, op_a, value_a, col_b, op_b, value_b):
    """
    Count days that satisfy Event A, Event B and both together.
    Returns a dictionary, or None if there is no usable data.
    """
    a_values = pd.to_numeric(df[col_a], errors="coerce")
    b_values = pd.to_numeric(df[col_b], errors="coerce")
    usable = a_values.notna() & b_values.notna()
    a_values, b_values = a_values[usable], b_values[usable]

    total = int(len(a_values))
    if total == 0:
        return None

    event_a = OPERATORS[op_a](a_values, value_a)
    event_b = OPERATORS[op_b](b_values, value_b)
    count_a = int(event_a.sum())
    count_b = int(event_b.sum())
    count_both = int((event_a & event_b).sum())

    return {
        "total": total,
        "count_a": count_a,
        "count_b": count_b,
        "count_both": count_both,
        "a_only": count_a - count_both,
        "b_only": count_b - count_both,
        "neither": total - count_a - count_b + count_both,
        "probability": count_both / total,
    }


# ---------------------------------------------------------------------------
# MODULE 2 - Discrete probability distribution (energy rating 1-10)
#   E(X) = sum of x * P(X = x)        Var(X) = E(X^2) - [E(X)]^2
# ---------------------------------------------------------------------------
def discrete_distribution(series, low=1, high=10):
    """Frequency table, expected value, variance and standard deviation."""
    values = pd.to_numeric(series, errors="coerce").dropna().round().astype(int)
    n = int(len(values))
    if n == 0:
        return None

    possible = list(range(low, high + 1))
    counts = values.value_counts().reindex(possible, fill_value=0)
    probabilities = counts / n

    x = np.array(possible, dtype=float)
    p = probabilities.to_numpy(dtype=float)

    expected = float(np.sum(x * p))              # E(X)
    expected_square = float(np.sum(x ** 2 * p))  # E(X^2)
    variance = max(expected_square - expected ** 2, 0.0)
    std = float(np.sqrt(variance))

    table = pd.DataFrame(
        {
            "Rating": possible,
            "Days": counts.to_numpy(dtype=int),
            "Probability": p,
            "x * P(x)": x * p,
        }
    )
    top = int(counts.max())
    return {
        "n": n,
        "table": table,
        "expected": expected,
        "expected_square": expected_square,
        "variance": variance,
        "std": std,
        "most_common": [int(v) for v in counts.index[counts == top]],
        "p_high": float(probabilities[probabilities.index >= 7].sum()),
    }


# ---------------------------------------------------------------------------
# MODULE 3 - Continuous probability distribution
# ---------------------------------------------------------------------------
def continuous_summary(series):
    """Mean, sample variance (n - 1) and standard deviation of one variable."""
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return None
    variance = float(np.var(x, ddof=1)) if n > 1 else float("nan")
    return {
        "n": n,
        "values": x,
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "variance": variance,
        "std": float(np.sqrt(variance)) if n > 1 else float("nan"),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def normal_fit_check(x):
    """
    Decide whether a normal (bell-curve) model is reasonable for the data
    using the Shapiro-Wilk test. Returns a dictionary with an explanation.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 8:
        return {"reasonable": False, "p_value": None,
                "reason": "Fewer than 8 records, so a bell-curve check is not reliable."}
    if np.std(x) == 0:
        return {"reasonable": False, "p_value": None,
                "reason": "All values are identical, so there is no bell-curve shape to fit."}
    p_value = float(stats.shapiro(x).pvalue)
    if p_value >= 0.05:
        return {"reasonable": True, "p_value": p_value,
                "reason": "The data does not show strong evidence against a bell-curve shape."}
    return {"reasonable": False, "p_value": p_value,
            "reason": "The data shows evidence of not being bell-shaped, so a normal fit may mislead."}


def density_curve(x, points=200):
    """Smooth density estimate (kernel density). Returns (grid, density) or None."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.std(x) == 0:
        return None
    try:
        kde = stats.gaussian_kde(x)
    except (np.linalg.LinAlgError, ValueError):
        return None
    spread = np.std(x, ddof=1)
    grid = np.linspace(x.min() - spread, x.max() + spread, points)
    return grid, kde(grid)


def empirical_cdf(x):
    """Cumulative distribution from the data itself: share of days <= each value."""
    xs = np.sort(np.asarray(x, dtype=float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def empirical_probability_between(x, low, high):
    """Share of recorded days with low <= value <= high (no assumptions)."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    count = int(np.sum((x >= low) & (x <= high)))
    return {"count": count, "total": len(x), "probability": count / len(x)}


def normal_probability_between(mean, std, low, high):
    """P(low <= X <= high) if X is normal(mean, std). None if std is not positive."""
    if std is None or not np.isfinite(std) or std <= 0:
        return None
    return float(stats.norm.cdf(high, mean, std) - stats.norm.cdf(low, mean, std))


# ---------------------------------------------------------------------------
# Shared helpers for modules 4 and 5
# ---------------------------------------------------------------------------
def _paired(x, y):
    """Keep only days where both variables have a value."""
    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(pd.Series(np.asarray(x)), errors="coerce"),
            "y": pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce"),
        }
    ).dropna()
    return pair["x"].to_numpy(dtype=float), pair["y"].to_numpy(dtype=float)


def _sums(xa, ya):
    """Return means and the three sums used by regression and correlation."""
    x_mean, y_mean = xa.mean(), ya.mean()
    sxx = float(np.sum((xa - x_mean) ** 2))
    syy = float(np.sum((ya - y_mean) ** 2))
    sxy = float(np.sum((xa - x_mean) * (ya - y_mean)))
    return x_mean, y_mean, sxx, syy, sxy


def _check_pair(xa, ya, x_name="X", y_name="Y"):
    """Return an error message if the pair cannot be analysed, otherwise None."""
    if len(xa) < MIN_RECORDS:
        return f"At least {MIN_RECORDS} days with both values are needed (you have {len(xa)})."
    _, _, sxx, syy, _ = _sums(xa, ya)
    if sxx <= 1e-12:
        return f"{x_name} has the same value on every day, so a relationship cannot be measured."
    if syy <= 1e-12:
        return f"{y_name} has the same value on every day, so a relationship cannot be measured."
    return None


# ---------------------------------------------------------------------------
# MODULE 4 - Regression lines
#   Y on X:  Y - Ybar = b_yx (X - Xbar),   b_yx = Sxy / Sxx
#   X on Y:  X - Xbar = b_xy (Y - Ybar),   b_xy = Sxy / Syy
# ---------------------------------------------------------------------------
def regression_analysis(x, y, x_name="X", y_name="Y"):
    xa, ya = _paired(x, y)
    message = _check_pair(xa, ya, x_name, y_name)
    if message:
        return {"ok": False, "message": message}

    x_mean, y_mean, sxx, syy, sxy = _sums(xa, ya)
    b_yx = sxy / sxx
    b_xy = sxy / syy
    r = sxy / np.sqrt(sxx * syy)
    return {
        "ok": True,
        "n": len(xa),
        "x": xa,
        "y": ya,
        "x_mean": float(x_mean),
        "y_mean": float(y_mean),
        "b_yx": float(b_yx),
        "b_xy": float(b_xy),
        "a_yx": float(y_mean - b_yx * x_mean),   # Y = a_yx + b_yx * X
        "a_xy": float(x_mean - b_xy * y_mean),   # X = a_xy + b_xy * Y
        "r": float(r),
        "r_squared": float(r ** 2),
    }


def predict_y_from_x(result, x_value):
    """Estimate Y using the regression of Y on X."""
    return result["y_mean"] + result["b_yx"] * (x_value - result["x_mean"])


def predict_x_from_y(result, y_value):
    """Estimate X using the regression of X on Y."""
    return result["x_mean"] + result["b_xy"] * (y_value - result["y_mean"])


# ---------------------------------------------------------------------------
# MODULE 5 - Pearson correlation coefficient
#   r = sum[(x - xbar)(y - ybar)] / sqrt( sum(x - xbar)^2 * sum(y - ybar)^2 )
# ---------------------------------------------------------------------------
def describe_correlation(r):
    """Return (direction, strength) in plain words for a Pearson r value."""
    size = abs(r)
    if size < 0.1:
        strength = "almost no"
    elif size < 0.3:
        strength = "weak"
    elif size < 0.5:
        strength = "moderate"
    elif size < 0.7:
        strength = "fairly strong"
    elif size < 0.9:
        strength = "strong"
    else:
        strength = "very strong"

    if size < 0.1:
        direction = "no clear direction"
    elif r > 0:
        direction = "positive"
    else:
        direction = "negative"
    return direction, strength


def pearson_correlation(x, y, x_name="X", y_name="Y"):
    xa, ya = _paired(x, y)
    message = _check_pair(xa, ya, x_name, y_name)
    if message:
        return {"ok": False, "message": message}

    x_mean, y_mean, sxx, syy, sxy = _sums(xa, ya)
    r = float(sxy / np.sqrt(sxx * syy))
    r = max(-1.0, min(1.0, r))  # guard against tiny rounding errors
    direction, strength = describe_correlation(r)
    return {
        "ok": True,
        "n": len(xa),
        "r": r,
        "direction": direction,
        "strength": strength,
        "x": xa,
        "y": ya,
        "x_mean": float(x_mean),
        "y_mean": float(y_mean),
        "b_yx": float(sxy / sxx),
        "a_yx": float(y_mean - (sxy / sxx) * x_mean),
    }
