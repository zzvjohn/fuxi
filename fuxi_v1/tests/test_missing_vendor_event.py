# -*- coding: utf-8 -*-
"""Pair tests for resolve_action_terms public_missing_vendor_event_v1 and
independent_missed_event."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(r"e:/quant")
sys.path.insert(0, str(ROOT))
from fuxi_v1.action_evidence import resolve_action_terms, independent_missed_event
from fuxi_v1.datasets import file_hash

MISS_TEXT = (
    "证券代码：001323 证券简称：慕思股份 公告编号：2025-035 "
    "慕思健康睡眠股份有限公司2024年年度权益分派实施公告 "
    "二、权益分派方案 "
    "公司2024年年度权益分派方案为：以公司现有总股本剔除已回购股份6,642,589股后的"
    "393,367,411股为基数，向全体股东每10股派10元人民币现金（含税），共计派发现金红利"
    "人民币393,367,411.00元，不送红股，不以公积金转增股本。 "
    "除权除息参考价格=权益分派股权登记日收盘价-按总股本折算每股现金红利="
    "权益分派股权登记日收盘价-0.9833939元/股。 "
    "三、股权登记日与除权除息日 "
    "本次权益分派股权登记日为：2025年6月6日，除权除息日为：2025年6月9日。 "
    "五、权益分派方法 "
    "1、本公司此次委托中国结算深圳分公司代派的A股股东现金红利将于2025年6月9日"
    "通过股东托管证券公司直接划入其资金账户。")

INDEX = {"ts_code": "001323.SZ", "ann_date": "20250603", "name": "",
         "title": "慕思股份：2024年年度权益分派实施公告",
         "url": "https://disc.static.szse.cn/download/miss.pdf",
         "rec_time": "2026-10-13T13:33:03+00:00"}


def write_source(dirpath, name, text):
    pdf = Path(dirpath) / (name + ".pdf")
    txt = Path(dirpath) / (name + ".txt")
    pdf.write_bytes(b"%PDF-1.5 test\n%%EOF")
    txt.write_text(text, encoding="utf-8")
    return {
        "url": INDEX["url"], "path": str(pdf.resolve()), "sha256": file_hash(pdf),
        "text_path": str(txt.resolve()), "text_sha256": file_hash(txt),
        "excerpt": text.replace("\n", "")[:4500], "role": "implementation",
        "publication_index": dict(INDEX),
    }


def make_judgment(sources, **over):
    judgment = {
        "instrument": "001323.XSHE", "ex_date": "2025-06-09",
        "type": "public_missing_vendor_event_v1",
        "record_date": "2025-06-06", "ann_date": "2025-06-03", "pay_date": "2025-06-09",
        "holder_cash": 1.0, "holder_shares": 0.0, "share_credit_date": None,
        "vendor_evidence_id": None, "reference_cash_deduction": 0.9833939,
        "sources": sources, "independent_assurance": False,
        "method": "archived_public_mirror_v1:missing_vendor_event",
    }
    judgment.update(over)
    return judgment


def test_missing_vendor_event_synthetic_term():
    with TemporaryDirectory() as tmp:
        source = write_source(tmp, "miss", MISS_TEXT)
        out = resolve_action_terms([], [], [make_judgment([source])])
    assert len(out) == 1, out
    term = out[0]
    assert term["status"] == "cash_terms_supported", term["status"]
    assert term["instrument"] == "001323.XSHE" and term["ex_date"] == "2025-06-09"
    assert term["gross_cash_per_share"] == 1.0
    assert term["share_distribution"] == 0.0
    assert term["reference_cash_deduction"] == 0.9833939
    assert term["vendor_event_row_missing"] is True
    assert term["record_date"] == "2025-06-06" and term["ann_date"] == "2025-06-03"
    assert term["pay_date"] == "2025-06-09"
    assert term["evidence_id"]


def test_rejects_vendor_rows_cover_ex_date():
    vendor = {"ts_code": "001323.SZ", "end_date": "20241231", "ann_date": "20250603",
              "div_proc": "实施", "stk_div": "0.0", "cash_div_tax": "1.0",
              "record_date": "20250606", "ex_date": "20250609", "pay_date": "20250609"}
    with TemporaryDirectory() as tmp:
        source = write_source(tmp, "miss", MISS_TEXT)
        try:
            resolve_action_terms([], [vendor], [make_judgment([source])])
        except Exception as exc:
            assert "already cover the missing-vendor ex-date" in str(exc), str(exc)
        else:
            raise AssertionError("vendor row covering the ex-date must be rejected")


def test_rejects_existing_base_term():
    base = {"instrument": "001323.XSHE", "ex_date": "2025-06-09",
            "status": "cash_terms_supported", "gross_cash_per_share": 1.0,
            "share_distribution": 0.0, "evidence_id": "base-evidence"}
    with TemporaryDirectory() as tmp:
        source = write_source(tmp, "miss", MISS_TEXT)
        try:
            resolve_action_terms([base], [], [make_judgment([source])])
        except Exception as exc:
            assert "must not bind an existing vendor term" in str(exc), str(exc)
        else:
            raise AssertionError("base term covering the ex-date must be rejected")


def test_rejects_missing_vendor_clock_conflict():
    with TemporaryDirectory() as tmp:
        source = write_source(tmp, "miss", MISS_TEXT)
        try:
            resolve_action_terms([], [], [make_judgment([source], pay_date="2025-06-07")])
        except Exception as exc:
            assert "clock conflict" in str(exc), str(exc)
        else:
            raise AssertionError("pay before ex must be rejected")


def test_rejects_share_distribution_without_credit():
    with TemporaryDirectory() as tmp:
        source = write_source(tmp, "miss", MISS_TEXT)
        try:
            resolve_action_terms([], [], [make_judgment([source], holder_shares=0.3)])
        except Exception as exc:
            assert "share settlement date missing" in str(exc), str(exc)
        else:
            raise AssertionError("shares without credit date must be rejected")


def test_rejects_archived_source_changed():
    with TemporaryDirectory() as tmp:
        source = write_source(tmp, "miss", MISS_TEXT)
        Path(source["text_path"]).write_text("tampered", encoding="utf-8")
        try:
            resolve_action_terms([], [], [make_judgment([source])])
        except Exception as exc:
            assert "source changed" in str(exc), str(exc)
        else:
            raise AssertionError("tampered archive must be rejected")


def test_rejects_adjudication_amount_mismatch():
    with TemporaryDirectory() as tmp:
        source = write_source(tmp, "miss", MISS_TEXT)
        try:
            resolve_action_terms([], [], [make_judgment([source], holder_cash=0.98)])
        except Exception as exc:
            assert "differs from archived terms" in str(exc), str(exc)
        else:
            raise AssertionError("adjudication amount mismatch must be rejected")


def test_independent_missed_event_accepted():
    before = {"tradestatus": "1", "close": "31.290000"}
    after = {"tradestatus": "1", "preclose": "30.310000", "close": "29.760000"}
    term = {"gross_cash_per_share": 1.0, "share_distribution": 0.0,
            "reference_cash_deduction": 0.9833939}
    assert independent_missed_event(before, after, {"2025-06-09"}, "2025-06-09", term)


def test_independent_missed_event_rejects():
    before = {"tradestatus": "1", "close": "31.290000"}
    after = {"tradestatus": "1", "preclose": "30.310000", "close": "29.760000"}
    term = {"gross_cash_per_share": 1.0, "share_distribution": 0.0,
            "reference_cash_deduction": 0.9833939}
    assert not independent_missed_event(before, after, set(), "2025-06-09", term)
    assert not independent_missed_event(before, after, {"2025-06-10"}, "2025-06-09", term)
    halted = dict(after, tradestatus="0")
    assert not independent_missed_event(before, halted, {"2025-06-09"}, "2025-06-09", term)
    drifted = dict(after, preclose="30.200000")
    assert not independent_missed_event(before, drifted, {"2025-06-09"}, "2025-06-09", term)
    shared = dict(term, share_distribution=0.3)
    assert not independent_missed_event(before, after, {"2025-06-09"}, "2025-06-09", shared)
    assert not independent_missed_event(None, after, {"2025-06-09"}, "2025-06-09", term)
    assert not independent_missed_event(before, None, {"2025-06-09"}, "2025-06-09", term)


def test_independent_missed_event_without_deduction_field():
    before = {"tradestatus": "1", "close": "31.290000"}
    after = {"tradestatus": "1", "preclose": "30.310000", "close": "29.760000"}
    term = {"gross_cash_per_share": 1.0, "share_distribution": 0.0}
    assert not independent_missed_event(before, after, {"2025-06-09"}, "2025-06-09", term)


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
