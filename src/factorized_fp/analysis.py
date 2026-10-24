"""Observed-checkpoint crossings and descriptive summaries."""

import math
import statistics


def crossing(history, metric, threshold, *, higher_is_better):
    """First recorded observation that reaches a threshold; no interpolation.

    Returns the reached observation, the immediately preceding observation, and
    their pair. For the lower-is-better metric the first record below the
    threshold is used; for accuracy the first record at or above it.
    """
    previous = None
    for row in history:
        value = row["validation"][metric]
        reached = value >= threshold if higher_is_better else value <= threshold
        if reached:
            return {"observed": row, "previous": previous,
                    "observed_seconds": row["cumulative_seconds"],
                    "previous_seconds": None if previous is None else previous["cumulative_seconds"]}
        previous = row
    return None


def summary(values):
    values = list(values)
    return {"mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values)}


def paired_ratio(candidate_seconds, reference_seconds):
    return reference_seconds / candidate_seconds


__all__ = ["crossing", "summary", "paired_ratio"]
