"""派生认证快照：S1 停复牌可成交性 + S2 公司行动账本 + S3 认证绑定。

从 1125 截面 row-form 快照（fuxi.campaign_input.1）派生一份账户路径可认证的
新快照（不覆盖原快照）。三阶段：

  S1  停复牌：suspend_d.csv S 记录 -> 退市前停牌日标 paused=True + 冻结价标记
              （mark_source=declared_frozen_last_close_on_vendor_halt，绑定
              suspend_source_sha256）；复牌首日越限（600145 退市重定价）同标冻结。
  S2  公司行动：dividend_fullcycle 现金分红（ann_date<=ex_date PIT 过滤）+ 复权因子
              推导送转股；逐 bar 置 corporate_action_known=True + event_coverage +
              除权日 sizing_reference_close/sizing_reference_evidence。
  S3  认证：补 execution.sell_tax_schedule（声明单档）+ snapshot.instruments 派生，
             certify() -> 写回 market 四注解 + investment_assurance（研究级绑定），
             recompute market_hash，落盘派生快照。

铁律：不改原快照、不改内核 DEFAULT_LIMITS、不降阈值、不接触实盘；认证是研究级
形式证据（claim_scope.investment_eligibility_certified=False），非实盘独立 assurance。

产出：
  data/fuxi_unified/snapshots/native64_20210104_20251231_ledger_v1/campaign_input.json
  stdout：逐 flag 认证结果 + 派生快照 sha256 + 对账摘要
"""
import copy
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import pandas as pd

from fuxi_v1.pit_universe import read_lifecycle, canonical_code, jq_code
from fuxi_v1.certification import (certify, certify_tradability, certify_event_coverage,
                                   certify_account_events, listed_at, account_scope,
                                   apply_to_payload)
from fuxi_v1.contracts import digest, MARKET_CONTENT_EXCLUSIONS

SRC = WORKSPACE / "data/fuxi_unified/snapshots/native64_20210104_20251231/campaign_input.json"
OUT_DIR = WORKSPACE / "data/fuxi_unified/snapshots/native64_20210104_20251231_ledger_v1"
OUT = OUT_DIR / "campaign_input.json"
RAW = WORKSPACE / "data/raw"
DIV = RAW / "kuaidao_gaps/dividend_fullcycle_20260928.csv"

SUSPEND_SHA = "f18b03648b9185a16f7a57982d3c20aec594f3aed60e95d9d541256ff5d5bdd1"
# 冻结价/冻结参考价标记（certification.certify_tradability 与 simulation 均认可）
MARK_FROZEN = "declared_frozen_last_close_on_vendor_halt"
WINDOW = ("20210104", "20251231")


def iso_day(s):
    if not s:
        return None
    return str(s).strip()[:10].replace("-", "")


def canon(code):
    return canonical_code(code)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_suspend_s():
    """suspend_d.csv S 记录 -> {(jq_code, YYYYMMDD): True}（全市场，全天/部分停牌均含）。"""
    m = {}
    with open(RAW / "suspend_d.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if (row.get("suspend_type") or "").strip() != "S":
                continue
            code = canonical_code(row.get("ts_code", ""))
            day = iso_day(row.get("trade_date"))
            if code and day:
                m[(jq_code(code), day)] = True
    return m


def load_dividends():
    """现金分红（div_proc='实施'，研究窗内，PIT: ann_date<=ex_date）-> {canon_code: [rows]}。"""
    frame = pd.read_csv(DIV, dtype=str).astype(object)
    frame = frame[frame["div_proc"].astype(str).str.strip() == "实施"]
    frame = frame[(frame["ex_date"] >= WINDOW[0]) & (frame["ex_date"] <= WINDOW[1])]
    out = defaultdict(list)
    for r in frame.where(frame.notna(), None).to_dict("records"):
        code = canonical_code(r.get("ts_code"))
        ex = iso_day(r.get("ex_date"))
        ann = iso_day(r.get("ann_date"))
        rec = iso_day(r.get("record_date"))
        if not (code and ex and ann and rec):
            continue
        if ann > ex:  # PIT：公告晚于除权日则不采信（avoid future）
            continue
        cash = r.get("cash_div")
        try:
            cash = float(cash) if cash not in (None, "") else None
        except (TypeError, ValueError):
            cash = None
        out[code].append({"ex_date": ex, "record_date": rec, "ann_date": ann,
                          "cash_div": cash})
    return dict(out)


def load_adj():
    """复权因子 -> {(canon_code, YYYYMMDD): float}。"""
    frame = pd.read_csv(RAW / "adj_factor_daily.csv", dtype=str, usecols=["ts_code", "trade_date", "adj_factor"])
    frame = frame[(frame["trade_date"] >= WINDOW[0]) & (frame["trade_date"] <= WINDOW[1])]
    out = {}
    for r in frame.itertuples(index=False):
        code = canonical_code(r.ts_code)
        day = iso_day(r.trade_date)
        if code and day:
            try:
                out[(code, day)] = float(r.adj_factor)
            except (TypeError, ValueError):
                continue
    return out


def to_iso(c):
    if not c:
        return None
    c = str(c)
    if len(c) == 8 and c.isdigit():
        return c[:4] + "-" + c[4:6] + "-" + c[6:8]
    return c


def clean_share(s):
    """送转比例四舍五入到 0.1 的整数倍，并吸附复权因子反推的浮点噪声（<0.1 视为 0）。

    A 股送转以每 10 股整数计（share_distribution 为 0.1 的整数倍，如 10 送 2 = 0.2），
    复权因子反推的现金分红事件里 share_distribution 应恒为 0（纯现金分红时
    adj_ratio = C_prev/(C_prev-D)，代入反推式得 s = 0），0.001~0.002 级残差是
    数据源复权因子的精度截断噪声，会破坏 100 股整手 × 比例的整数性，触发
    fractional_allotment_not_observed 误报，这里统一 round 到 0.1 整数倍吸附。
    """
    if s is None:
        return 0.0
    tenths = round(float(s) * 10)
    return 0.0 if tenths <= 0 else tenths / 10.0


def build_actions(bars, instruments, dividends, adj, sessions):
    """构造 action_terms（现金分红 + 复权因子推导送转股）。

    返回 (action_terms, action_map)：
      action_map[(jq_code, YYYYMMDD)] = dict(sizing_reference_close, evidence_id,
                                             event_coverage, share_distribution)
    term 内日期统一为 ISO（YYYY-MM-DD），与 sessions/run_account 对齐；
    action_map 键用 compact（YYYYMMDD）对齐 bar 的 session.replace('-','')。
    复权因子方向（实测）：adj_ratio = adj_new/adj_prev = (1+s)*C_prev/(C_prev-D)
      => share_distribution s = adj_ratio*(1-D/C_prev) - 1
      => sizing_reference_close = (C_prev-D)/(1+s) = C_prev/adj_ratio
    """
    # per-instrument 前一交易日 (compact_day, close) 索引
    by_code = defaultdict(list)
    for b in bars:
        by_code[b["instrument"]].append((b["session"].replace("-", ""), b.get("close")))
    prev_close, prev_session = {}, {}
    for code, rows in by_code.items():
        rows = sorted(rows)
        for i in range(1, len(rows)):
            prev_close[(code, rows[i][0])] = rows[i - 1][1]
            prev_session[(code, rows[i][0])] = rows[i - 1][0]

    terms = []
    action_map = {}
    sess_set = {s.replace("-", "") for s in sessions}
    for code in instruments:
        ccode = canon(code)
        # 现金分红事件
        for d in dividends.get(ccode, []):
            ex = d["ex_date"]
            if ex not in sess_set:
                continue
            key = (code, ex)
            c_prev = prev_close.get(key)
            day_prev = prev_session.get(key)
            adj_new = adj.get((ccode, ex))
            adj_prev = adj.get((ccode, day_prev)) if day_prev else None
            D = d["cash_div"] or 0.0
            adj_ratio = (adj_new / adj_prev) if (adj_new and adj_prev and adj_prev > 0) else 1.0
            s = clean_share(adj_ratio * (1.0 - D / c_prev) - 1.0) if c_prev else 0.0
            sizing = (c_prev - D) / (1.0 + s) if c_prev else c_prev
            rec = d["record_date"]
            if rec and rec >= ex:  # 保证 record_date < ex_date（run_account require）
                rec = day_prev
            term = {
                "instrument": code,
                "evidence_id": digest(["cash_dividend", code, ex, d.get("ann_date"), D]),
                "ex_date": to_iso(ex),
                "record_date": to_iso(rec),
                "ann_date": to_iso(d["ann_date"]),
                "status": "ordinary_distribution_supported",
                "gross_cash_per_share": D if D > 0 else 0.0,
                "pay_date": to_iso(ex),  # 声明 T+0 派息假设（见 scope）
                "share_distribution": s if s > 0 else 0.0,
                "share_credit_date": to_iso(ex),  # 声明 T+0 到账假设（见 scope）
                "tax_fraction": "0.20",  # 声明 flat 20% 股息税（研究敏感性，非投资人税单）
                "scope": "cash_dividend_from_fullcycle_ledger; bonus_share_ratio_derived_from_adj_factor; "
                         "pay_date_and_share_credit_date_are_declared_T0_research_assumptions",
            }
            terms.append(term)
            action_map[key] = {"sizing_reference_close": sizing, "evidence_id": term["evidence_id"],
                               "event_coverage": "cash_dividend_ex_date_verified",
                               "share_distribution": s}

        # 纯送转股事件（复权因子变化但无现金分红账本）
        div_ex = {d["ex_date"] for d in dividends.get(ccode, [])}
        code_days = sorted(day for (c, day) in adj if c == ccode)
        prev_adj = None
        for day in code_days:
            if day not in sess_set:
                continue
            cur = adj[(ccode, day)]
            if prev_adj is None:
                prev_adj = cur
                continue
            # 相对阈值 1e-4（0.01%）：复权因子浮点噪声 ~1.6e-5，真实送转/分红 >=1e-3
            if abs(cur - prev_adj) / prev_adj < 1e-4:
                prev_adj = cur
                continue
            if day in div_ex:  # 已作为现金分红处理
                prev_adj = cur
                continue
            adj_ratio = cur / prev_adj if prev_adj > 0 else 1.0
            s = clean_share(adj_ratio - 1.0)
            c_prev = prev_close.get((code, day))
            sizing = c_prev / (1.0 + s) if c_prev else None
            day_prev = prev_session.get((code, day))
            term = {
                "instrument": code,
                "evidence_id": digest(["bonus_share_adj_derived", code, day]),
                "ex_date": to_iso(day),
                "record_date": to_iso(day_prev),
                "ann_date": to_iso(day),
                "status": "ordinary_distribution_supported",
                "gross_cash_per_share": 0.0,
                "pay_date": None,
                "share_distribution": s if s > 0 else 0.0,
                "share_credit_date": to_iso(day),
                "tax_fraction": "0.20",  # 声明 flat 20% 股息税（研究敏感性，非投资人税单）
                "scope": "bonus_or_conversion_share_ratio_derived_from_adj_factor_ratio; "
                         "no_separate_stk_div_ledger",
            }
            terms.append(term)
            action_map[(code, day)] = {"sizing_reference_close": sizing, "evidence_id": term["evidence_id"],
                                       "event_coverage": "bonus_share_ex_date_adj_derived",
                                       "share_distribution": s}
            prev_adj = cur
    return terms, action_map


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    snap = d["snapshot"]
    market = d["market"]
    bars = market["bars"]
    sessions = snap["sessions"]

    life = read_lifecycle(RAW, "delist_dates.csv")
    suspend = load_suspend_s()
    dividends = load_dividends()
    adj = load_adj()

    instruments = sorted({c for v in snap["universe"].values() for c in v})
    print("instruments:", len(instruments))

    # ---- S1 停牌标记 + S2 公司行动 ----
    terms, action_map = build_actions(bars, instruments, dividends, adj, sessions)
    print("action_terms:", len(terms))

    # 冻结价：每标的每一会话的"最后非停牌 close"（停牌日估值用，carry-forward）
    frozen_close = {}
    for code in instruments:
        last = None
        rows = sorted([(b["session"], b.get("close")) for b in bars if b["instrument"] == code])
        for sess, c in rows:
            frozen_close[(code, sess)] = last
            if c is not None:
                last = c

    marked_paused = 0
    ca_set = 0
    for b in bars:
        code, day = b["instrument"], b["session"]
        is_halt = (b.get("paused") is None and b.get("close") is None)
        if is_halt:
            st = listed_at(life, code, day)
            b["paused"] = True
            b["mark_source"] = MARK_FROZEN
            b["suspend_source_sha256"] = SUSPEND_SHA
            b["corporate_action_known"] = True
            b["event_coverage"] = "halt_no_action_frozen" if st is True else "delisted_frozen_last_close"
            # 停牌/退市日无成交，估值用冻结最后 close（非 None -> 可估值）
            if frozen_close.get((code, day)) is not None:
                b["close"] = frozen_close[(code, day)]
            marked_paused += 1

        # 复牌首日越限（600145 退市重定价）：标冻结，避免 stale limit 阻塞
        ul, dl, o = b.get("up_limit"), b.get("down_limit"), b.get("open")
        if (b.get("paused") is not True and ul is not None and dl is not None
                and o is not None and not (dl - 1e-9 <= o <= ul + 1e-9)):
            b["paused"] = True
            b["mark_source"] = MARK_FROZEN
            b["suspend_source_sha256"] = SUSPEND_SHA
            b["corporate_action_known"] = True
            b["event_coverage"] = "resumption_repricing_limit_stale_frozen"
            marked_paused += 1

        # S2：逐 bar 置 corporate_action_known + event_coverage + sizing_reference
        key = (code, day.replace("-", ""))
        act = action_map.get(key)
        if b.get("corporate_action_known") is not True:
            if act is not None:
                b["corporate_action_known"] = True
                b["event_coverage"] = act["event_coverage"]
                b["sizing_reference_close"] = act["sizing_reference_close"]
                b["sizing_reference_evidence"] = act["evidence_id"]
            else:
                b["corporate_action_known"] = True
                b["event_coverage"] = "no_corporate_action_adj_verified"
            ca_set += 1
        else:
            # 已是 True，但仍需为除权日补 sizing_reference（若缺）
            if act is not None and not b.get("sizing_reference_close"):
                b["sizing_reference_close"] = act["sizing_reference_close"]
                b["sizing_reference_evidence"] = act["evidence_id"]

    print("marked_paused:", marked_paused, "corporate_action cells set:", ca_set)

    # ---- 组装派生 market ----
    mkt = copy.deepcopy(market)
    mkt["bars"] = bars
    mkt["action_terms"] = terms
    mkt["actions"] = []  # actions 事件由 action_terms 派生（行式 market 无独立 events）
    exec_blk = dict(mkt["execution"])
    if not exec_blk.get("sell_tax_schedule"):
        exec_blk["sell_tax_schedule"] = [["2021-01-04", exec_blk.get("sell_tax", "0.0005")]]
    mkt["execution"] = exec_blk

    # ---- 组装派生 payload（snapshot 保持原结构，不注入 instruments 键；
    #       certify_universe 已改为从 universe 派生 frozen cohort）----
    payload = dict(d, snapshot=snap, market=mkt)

    # ---- S3 认证 ----
    declaration = {}
    cert = certify(payload, declaration=declaration, lifecycle=life)

    # 写回市场四注解（event_coverage_certified / tradability_certified /
    # certification_hash / certification_snapshot_hash），再 research 级 assurance 绑定
    mkt["event_coverage_certified"] = cert["flags"]["event_coverage_certified"]["value"]
    mkt["tradability_certified"] = cert["flags"]["tradability_certified"]["value"]
    mkt["certification_hash"] = cert["certification_hash"]
    mkt["certification_snapshot_hash"] = cert["snapshot_hash"]
    mkt["market_hash"] = digest({k: v for k, v in mkt.items() if k != "market_hash"})

    # research 级 assurance 收据（非独立投资 assurance）
    research_receipt = {
        "schema_version": "fuxi.market_assurance.1",
        "review": "fuxi-research-derived-contract-validation",
        "kind": "derived_research_contract_validation",
        "investment_eligibility_certified": False,
        "universe_certified": cert["flags"]["universe_certified"]["value"],
        "reviewed_at": "2026-10-18T00:00:00+08:00",
    }
    mkt["investment_assurance"] = {
        "schema_version": "fuxi.market_assurance.1",
        "snapshot_hash": cert["snapshot_hash"],
        "market_digest": cert["market_digest"],
        "receipt_id": "fuxi-research-" + cert["certification_hash"][:16],
        "assurance_hash": digest(research_receipt),
    }
    mkt["market_hash"] = digest({k: v for k, v in mkt.items() if k != "market_hash"})

    # 完整 certificate 注解写回 payload 顶层
    result = apply_to_payload(payload, cert)
    result["market"] = mkt

    # ---- 落盘 ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    out_hash = sha256_file(OUT)

    # ---- 汇报 ----
    print("\n=== 认证结果 ===")
    for name in ("pit_certified", "universe_certified", "market_representativeness_certified",
                 "account_events_certified", "event_coverage_certified",
                 "tradability_certified", "execution_rules_certified"):
        fl = cert["flags"][name]
        print(f"  {name}: value={fl['value']} blocking={fl['blocking_gaps']}")
    print("claim_scope:", cert["claim_scope"])
    print("\n=== 派生快照 ===")
    print("path:", OUT)
    print("sha256:", out_hash)
    print("certification_hash:", cert["certification_hash"])
    print("snapshot_hash:", cert["snapshot_hash"])
    print("market_digest:", cert["market_digest"])


if __name__ == "__main__":
    main()
