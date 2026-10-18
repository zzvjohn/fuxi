"""Read-only prospective-pool counterfactual for the frozen model comparison.

The 2026-09-24 script and its protocol hash remain untouched. This helper
compares selection identities only; unavailable future returns stay missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compare_prospective_vs_hindsight_pool(
    frame: pd.DataFrame, score: pd.Series, future_label: pd.Series, *, top_n: int = 80
) -> dict:
    """Compare Top N on a signal-date pool to the old label-complete pool."""
    if frame.index.has_duplicates or frame.index.name != "ts_code":
        raise ValueError("signal-date pool needs unique ts_code index")
    if not frame.index.equals(score.index):
        raise ValueError("scores must be aligned with signal-date pool")
    if top_n < 1:
        raise ValueError("top_n must be positive")

    def select(f: pd.DataFrame, s: pd.Series) -> list[str]:
        work = f.assign(score=s).reset_index()
        ordered = work.sort_values(["score", "amount", "ts_code"],
                                   ascending=[False, False, True]).head(top_n)
        return list(ordered.ts_code)

    label = pd.to_numeric(future_label.reindex(frame.index), errors="coerce")
    future_finite = np.isfinite(label.to_numpy(dtype=float))
    prior = select(frame.loc[future_finite], score.loc[future_finite])
    prospective = select(frame, score)
    missing_selected = [code for code in prospective if not bool(np.isfinite(label.loc[code]))]
    return {"signal_pool_n": int(len(frame)), "future_label_finite_n": int(future_finite.sum()),
            "hindsight_top_codes": prior, "prospective_top_codes": prospective,
            "prospective_unlabelled_top_codes": missing_selected,
            "top_overlap_n": int(len(set(prior) & set(prospective)))}
