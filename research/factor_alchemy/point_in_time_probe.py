"""Read-only future-row perturbation probe for an existing formula evaluator.

This tests the evaluator's actual behavior on one panel. Passing the probe is
necessary evidence of causality, not proof for every possible input or field.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def probe_future_row_invariance(
    panel: pd.DataFrame,
    cutoff: object,
    evaluate: Callable[[pd.DataFrame], pd.DataFrame],
) -> dict:
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        raise ValueError("a nonempty time-indexed DataFrame is required")
    if panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError("time index must be unique and sorted")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in panel.dtypes):
        raise ValueError("probe requires numeric factor inputs")
    past = panel.index <= cutoff
    future = panel.index > cutoff
    if not past.any() or not future.any():
        raise ValueError("both historical and future rows are required")

    baseline = evaluate(panel.copy(deep=True))
    if not isinstance(baseline, pd.DataFrame) or not baseline.index.equals(panel.index):
        raise ValueError("evaluator must return a time-aligned DataFrame")
    for multiplier in (0.01, 100.0):
        changed = panel.copy(deep=True)
        changed.loc[future, :] = changed.loc[future, :] * multiplier + multiplier
        observed = evaluate(changed)
        if not isinstance(observed, pd.DataFrame) or not observed.index.equals(panel.index):
            raise ValueError("evaluator output lost time alignment")
        if not observed.columns.equals(baseline.columns):
            raise ValueError("evaluator output changed its symbol set")
        left = baseline.loc[past].to_numpy(dtype=float)
        right = observed.loc[past].to_numpy(dtype=float)
        same = np.isclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True)
        if not bool(same.all()):
            positions = np.argwhere(~same)
            return {"status": "reject", "code": "future_rows_change_historical_signal",
                    "historic_rows": int(past.sum()), "future_rows": int(future.sum()),
                    "changed_historical_cells": int(len(positions)),
                    "first_changed_date": str(panel.index[np.flatnonzero(past)[positions[0, 0]]])}
    return {"status": "probe_pass", "code": None,
            "historic_rows": int(past.sum()), "future_rows": int(future.sum()),
            "changed_historical_cells": 0}
