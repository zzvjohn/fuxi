"""CSI300 / CSI500 benchmarks for E3 absolute-excess reporting.

The E3 gate historically compared a candidate only against the frozen straw-man
baseline ``rank(return(close,20))``. That baseline collapses (~-70% on the
2021-2025 research window), so a positive paired increment can coexist with a
negative absolute return. From 2026-10-19 E3 additionally reports absolute
excess versus CSI300 and CSI500, and requires a positive excess on both indices
whenever the candidate's absolute return is non-positive.

Index data is read from ``data/raw/index_000300_daily.csv`` and
``data/raw/index_000905_daily.csv``. The file hash and coverage are embedded in
the E3 metrics so the benchmark is auditable; when the files are unavailable the
E3 decision falls back to the historical paired-only gate and records
``benchmark_status='unavailable'`` (never silently invents data).
"""
import csv
import hashlib
import os
from pathlib import Path

INDEX_FILES = {
    "csi300": ("data/raw/index_000300_daily.csv", "000300.SH"),
    "csi500": ("data/raw/index_000905_daily.csv", "000905.SH"),
}
_CACHE = {}
MIN_COVERAGE = 0.98


def _compact(date):
    return str(date or "").replace("-", "").strip()


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def discover_root(start=None):
    if os.environ.get("FUXI_BENCHMARK_ROOT"):
        return Path(os.environ["FUXI_BENCHMARK_ROOT"]).resolve()
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / INDEX_FILES["csi300"][0]).is_file():
            return candidate
    return here


def load_index(name, root=None):
    root = Path(root).resolve() if root else discover_root()
    rel, code = INDEX_FILES[name]
    path = root / rel
    key = (str(path), _file_sha(path) if path.is_file() else None)
    if key in _CACHE:
        return _CACHE[key]
    if not path.is_file():
        value = {"status": "unavailable", "path": str(path), "sha256": None, "closes": {}}
        _CACHE[key] = value
        return value
    closes = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("ts_code") and row["ts_code"] != code:
                continue
            day = _compact(row.get("trade_date") or row.get("date"))
            close = row.get("close")
            if day and close not in (None, ""):
                try:
                    closes[day] = float(close)
                except ValueError:
                    continue
    value = {"status": "available" if closes else "unavailable", "path": str(path),
             "sha256": key[1], "closes": closes}
    _CACHE[key] = value
    return value


def _summarise(name, sessions, candidate_nav, candidate_returns, index):
    closes = index.get("closes") or {}
    if not closes or not sessions:
        return {"status": index.get("status", "unavailable")}
    compact = [_compact(d) for d in sessions]
    present = [c for c in compact if c in closes]
    coverage = len(present) / len(compact) if compact else 0.0
    if coverage < MIN_COVERAGE:
        return {"status": "insufficient_coverage", "coverage": round(coverage, 4)}
    base_day = compact[0]
    if base_day not in closes:
        earlier = [d for d in closes if d < base_day]
        if not earlier:
            return {"status": "insufficient_coverage", "coverage": round(coverage, 4)}
        base_day = max(earlier)
    total = closes[compact[-1]] / closes[base_day] - 1
    bench_returns = []
    for prev, cur in zip(compact, compact[1:]):
        if prev in closes and cur in closes and closes[prev] > 0:
            bench_returns.append(closes[cur] / closes[prev] - 1)
    excess_daily = [a - b for a, b in zip(candidate_returns, bench_returns)]
    excess_annual = (sum(excess_daily) / len(excess_daily) * 252) if excess_daily else None
    candidate_total = candidate_nav[-1] / candidate_nav[0] - 1 if len(candidate_nav) >= 2 else None
    return {
        "status": "available",
        "coverage": round(coverage, 4),
        "benchmark_total_return": total,
        "candidate_total_return": candidate_total,
        "excess": (candidate_total - total) if candidate_total is not None else None,
        "excess_daily_mean": (sum(excess_daily) / len(excess_daily)) if excess_daily else None,
        "excess_annualized": excess_annual,
        "sha256": index.get("sha256"),
    }


def absolute_excess(candidate_daily, root=None):
    """Cumulative and annualised excess of the candidate account vs CSI300/CSI500."""
    sessions, nav = [], []
    for row in candidate_daily:
        try:
            nav.append(float(row["nav"]))
        except (KeyError, TypeError, ValueError):
            continue
        sessions.append(row.get("session"))
    if len(nav) < 2:
        return {"status": "data_insufficient"}
    returns = [b / a - 1 for a, b in zip(nav, nav[1:])]
    root = Path(root).resolve() if root else discover_root()
    out = {"status": "available", "root": str(root)}
    for name in INDEX_FILES:
        out[name] = _summarise(name, sessions, nav, returns, load_index(name, root))
    if all(out[name].get("status") != "available" for name in INDEX_FILES):
        out["status"] = "unavailable"
    elif any(out[name].get("status") != "available" for name in INDEX_FILES):
        out["status"] = "partial"
    return out


def excess_gate(excess, candidate_total, require_on_nonpositive=True):
    """Gate: non-positive absolute return requires positive excess on both indices.

    Returns (ok, reason). When benchmarks are unavailable the historical
    behaviour is preserved (ok=True) and the reason names the missing evidence.
    """
    status = (excess or {}).get("status")
    if status not in ("available", "partial"):
        return True, "benchmark_unavailable_paired_only"
    if not require_on_nonpositive or candidate_total is not None and candidate_total > 0:
        return True, "absolute_return_positive"
    parts = [excess.get("csi300") or {}, excess.get("csi500") or {}]
    if any(p.get("status") != "available" for p in parts):
        return False, "benchmark_evidence_incomplete"
    if all((p.get("excess") or 0) > 0 for p in parts):
        return True, "absolute_return_nonpositive_but_excess_positive"
    return False, "absolute_return_nonpositive_without_excess"
