# -*- coding: utf-8 -*-
"""Pair tests for resolve_action_terms public_adjusted_final_terms_v1 branch."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(r"e:/quant")
sys.path.insert(0, str(ROOT))
from fuxi_v1.action_evidence import resolve_action_terms
from fuxi_v1.datasets import file_hash

VENDOR = {
    "ts_code": "003001.SZ", "end_date": "20241231", "ann_date": "20250418",
    "div_proc": "实施", "stk_div": "0.3992772", "stk_bo_rate": None,
    "stk_co_rate": "0.3992772", "cash_div": "0.2814904", "cash_div_tax": "0.2814904",
    "record_date": "20250704", "ex_date": "20250707", "pay_date": "20250707",
    "div_listdate": "20250707", "imp_ann_date": "20250705",
}
VENDOR2 = dict(VENDOR, imp_ann_date="20250630")


def make_base():
    return {
        "instrument": "003001.XSHE", "ex_date": "2025-07-07",
        "status": "conflicting_dividend_rows", "record_date": "2025-07-04",
        "ann_date": "2025-07-05", "pay_date": "2025-07-07",
        "gross_cash_per_share": 0.2814904, "share_distribution": 0.3992772,
        "evidence_id": "base-evidence",
    }


def make_judgment(sources, **over):
    judgment = {
        "instrument": "003001.XSHE", "ex_date": "2025-07-07",
        "type": "public_adjusted_final_terms_v1",
        "record_date": "2025-07-04", "ann_date": "2025-06-30",
        "adjustment_ann_date": "2025-07-05", "pay_date": "2025-07-07",
        "company_share_distribution": 0.3992772, "holder_cash": 0.2814904,
        "holder_shares": 0.3992772, "share_credit_date": "2025-07-07",
        "vendor_evidence_id": "base-evidence",
        "final_cash_per_10": 2.814904, "final_shares_per_10": 3.992772,
        "sources": sources, "independent_assurance": False,
        "method": "archived_public_mirror_v1:adjusted_final_terms",
    }
    judgment.update(over)
    return judgment


def write_source(dirpath, name, role, text, *, url="https://mirror.example/ann.pdf"):
    pdf = Path(dirpath) / (name + ".pdf")
    txt = Path(dirpath) / (name + ".txt")
    pdf.write_bytes(b"%PDF-1.5 test\n%%EOF")
    txt.write_text(text, encoding="utf-8")
    return {
        "url": url, "path": str(pdf.resolve()), "sha256": file_hash(pdf),
        "text_path": str(txt.resolve()), "text_sha256": file_hash(txt),
        "excerpt": "archived announcement text", "role": role,
    }


IMPL_TEXT = "003001 2024年度权益分派实施公告 每10股派2.816137元 每10股转增3.994521股 登记日2025-07-04 除权除息日2025-07-07"
ADJ_TEXT = ("003001 关于调整2024年度利润分配每股现金红利分配比例及每股转增股数的公告 "
            "调整前每10股派2.816137元 每10股转增3.994521股 "
            "调整后每10股派2.814904元 每10股转增3.992772股 落款2025-07-04")


def test_adjusted_final_terms_supported():
    with TemporaryDirectory() as tmp:
        impl = write_source(tmp, "impl", "implementation", IMPL_TEXT)
        adj = write_source(tmp, "adj", "adjusted_final_terms", ADJ_TEXT)
        out = resolve_action_terms([make_base()], [VENDOR, VENDOR2],
                                   [make_judgment([impl, adj])])
    assert len(out) == 1, out
    term = out[0]
    assert term["status"] == "ordinary_distribution_supported", term["status"]
    assert term["gross_cash_per_share"] == 0.2814904
    assert term["share_distribution"] == 0.3992772
    assert term["share_credit_date"] == "2025-07-07"
    assert term["record_date"] == "2025-07-04" and term["ann_date"] == "2025-06-30"
    assert "adjusted-final-terms" in term["settlement_scope"]


def test_rejects_vendor_final_amount_mismatch():
    with TemporaryDirectory() as tmp:
        impl = write_source(tmp, "impl", "implementation", IMPL_TEXT)
        adj = write_source(tmp, "adj", "adjusted_final_terms", ADJ_TEXT)
        judgment = make_judgment([impl, adj], holder_cash=0.2814)
        try:
            resolve_action_terms([make_base()], [VENDOR], [judgment])
        except Exception as exc:
            assert "final amounts differ" in str(exc), str(exc)
        else:
            raise AssertionError("vendor amount mismatch must be rejected")


def test_rejects_missing_adjustment_archive():
    with TemporaryDirectory() as tmp:
        impl = write_source(tmp, "impl", "implementation", IMPL_TEXT)
        dup = dict(impl, role="implementation")
        try:
            resolve_action_terms([make_base()], [VENDOR], [make_judgment([impl, dup])])
        except Exception as exc:
            assert "implementation and adjustment archives" in str(exc), str(exc)
        else:
            raise AssertionError("dual-role requirement must be enforced")


def test_rejects_adjustment_clock_conflict():
    with TemporaryDirectory() as tmp:
        impl = write_source(tmp, "impl", "implementation", IMPL_TEXT)
        adj = write_source(tmp, "adj", "adjusted_final_terms", ADJ_TEXT)
        judgment = make_judgment([impl, adj], adjustment_ann_date="2025-07-08")
        try:
            resolve_action_terms([make_base()], [VENDOR], [judgment])
        except Exception as exc:
            assert "adjustment clock conflict" in str(exc), str(exc)
        else:
            raise AssertionError("adjustment after ex-date must be rejected")


def test_rejects_implementation_archive_with_final_amounts():
    with TemporaryDirectory() as tmp:
        impl = write_source(tmp, "impl", "implementation", IMPL_TEXT + " 2.814904")
        adj = write_source(tmp, "adj", "adjusted_final_terms", ADJ_TEXT)
        try:
            resolve_action_terms([make_base()], [VENDOR], [make_judgment([impl, adj])])
        except Exception as exc:
            assert "predate final amounts" in str(exc), str(exc)
        else:
            raise AssertionError("implementation archive leaking final amounts must be rejected")


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception:
            failed += 1
            print("FAIL", fn.__name__)
            traceback.print_exc()
    print("total:", len(tests), "failed:", failed)
    sys.exit(1 if failed else 0)
