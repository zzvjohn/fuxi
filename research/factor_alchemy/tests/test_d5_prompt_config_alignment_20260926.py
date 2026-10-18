"""D5 定向测试：提示词/config 语义对齐（2026-09-26）。

契约（reports/fuxi_alpha_generation_overhaul_20260926.md §D5）:
  1. 提示词不得再承诺系统不执行的冻结/冷却
     （llm_generator 宪法条款 3 + ralph_loop system_prompt 注入点）
  2. config 旧名 budget_ledger_enabled 不存在；新名
     budget_ledger_monitor_enabled 默认 False（监控未接线 = 诚实态）
  3. budget_ledger.get_ledger() 读新键且**放行语义不变**（未接线不得变成阻塞）
  4. direction_campaign 留痕语义：resolve(缺 public_improved) 永不拦截（非门禁）
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FA = ROOT / "research" / "factor_alchemy"
sys.path.insert(0, str(FA))

LLM_SRC = (FA / "llm_generator.py").read_text(encoding="utf-8")
RALPH_SRC = (FA / "ralph_loop.py").read_text(encoding="utf-8")


class D5AlignmentTests(unittest.TestCase):
    def test_prompt_promises_removed_and_replaced(self):
        # 旧承诺必须消失（两处注入点；用原句断言，避免误伤 D5 留痕注释）
        self.assertNotIn("连续落空会被冻结冷却", LLM_SRC)
        self.assertNotIn("每个探索方向有尝试预算", LLM_SRC)
        self.assertNotIn("探索方向有尝试预算", RALPH_SRC)
        # 新表述：只承诺系统真的在做的（尝试记录/评审追溯）
        self.assertIn("【方向纪律】", LLM_SRC)
        self.assertIn("探索方向有尝试记录(供评审追溯)", RALPH_SRC)

    def test_config_rename_and_honest_default(self):
        from config import V06_EXPERIMENTAL as V06
        self.assertNotIn("budget_ledger_enabled", V06)
        self.assertIn("budget_ledger_monitor_enabled", V06)
        self.assertFalse(V06["budget_ledger_monitor_enabled"])
        # 方向战役保持只记账本通道（True 的含义已在 config 注释中标注）
        self.assertTrue(V06["direction_campaign_enabled"])

    def test_budget_ledger_reads_new_key_and_still_allows(self):
        from budget_ledger import get_ledger
        lg = get_ledger()
        self.assertFalse(lg.enabled)
        ok, reason = lg.can_generate("reversal", 1)
        self.assertTrue(ok)  # 放行语义不变（未接线 != 阻塞）
        self.assertIn(reason, ("BUDGET_DISABLED", "OK"))

    def test_ralph_loop_uses_new_key_only(self):
        self.assertIn('"budget_ledger_monitor_enabled"', RALPH_SRC)
        self.assertNotIn('get("budget_ledger_enabled"', RALPH_SRC)

    def test_campaign_never_blocks_without_public_improved(self):
        # D5 留痕的结构事实：无 public_improved 输入 ⇒ resolve 只记账本、永不拦截。
        # 用临时账本；不接触生产 campaign_ledger.jsonl。
        from direction_campaign import DirectionCampaign
        tmp = Path(tempfile.mkdtemp()) / "c.jsonl"
        dc = DirectionCampaign(ledger_path=tmp, enabled=True)
        for _ in range(5):
            d = dc.resolve("资金流")
            self.assertTrue(d["allow"])
            self.assertEqual(d["mode"], "CAMPAIGN")
        self.assertFalse(dc._is_cooled("资金流", "default"))


if __name__ == "__main__":
    unittest.main()
