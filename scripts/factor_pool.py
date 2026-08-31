"""
XQuant 因子假设实验 — 因子库
=============================
通过 Ch9 周频因子假设实验验证的因子纳入此库，
自动注册到 FA 的因子系统中。

原则: 节奏比努力重要 — 每周检验一批, 通过的纳入, 未通过的标记

创建: 2026-06-14
"""

from pathlib import Path
import pandas as pd

POOL_CSV = Path(__file__).parent.parent / 'data' / 'passed_factor_pool.csv'

# 当前已通过因子 (截至2026-06-14, 第1-2轮)
PASSED_FACTORS = [
    {
        'name': 'gap_up',
        'label': '跳空缺口持续性',
        'round': 1,
        'date': '2026-06-14',
        'icir': 0.437,
        'ic_mean': 0.0402,
        '+ic_pct': 0.673,
        'category': 'momentum',
        'hypothesis': '近期出现向上跳空缺口的股票有持续动量',
        'logic': '市场微观结构: 缺口反映信息冲击, 市场需要时间消化',
        'formula': 'open_p / close_p.shift(1) - 1',
        'direction': 'long',
        'status': 'candidate',  # candidate / integrated / retired
    },
    {
        'name': 'panic_selling',
        'label': '恐慌抛售代理',
        'round': 2,
        'date': '2026-06-14',
        'icir': 0.454,
        'ic_mean': 0.0553,
        '+ic_pct': 0.760,
        'category': 'behavioral',
        'hypothesis': '高成交+价格下跌的股票短期可能形成底部(恐慌盘出清)',
        'logic': '行为金融: 恐慌抛售+高换手=筹码换手充分, 短期修复概率高',
        'formula': '-(volume_p.rolling(3).mean() * (close_p.pct_change(3) < 0).astype(float).replace(0, NaN))',
        'direction': 'long',
        'status': 'candidate',
    },
]


def get_pool_df():
    """获取当前因子库"""
    if POOL_CSV.exists():
        existing = pd.read_csv(POOL_CSV)
        # 合并新通过因子
        new_names = [f['name'] for f in PASSED_FACTORS]
        existing_names = existing['name'].tolist() if len(existing) > 0 else []
        for f in PASSED_FACTORS:
            if f['name'] not in existing_names:
                existing = pd.concat([existing, pd.DataFrame([f])], ignore_index=True)
        return existing
    else:
        return pd.DataFrame(PASSED_FACTORS)


def save_pool():
    """保存因子库到 CSV"""
    df = get_pool_df()
    POOL_CSV.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(POOL_CSV, index=False, encoding='utf-8-sig')
    return df


def add_factor(name, label, round_num, icir, ic_mean, ic_pct, category,
               hypothesis, logic, formula, direction='long'):
    """手动添加通过因子"""
    global PASSED_FACTORS
    PASSED_FACTORS.append({
        'name': name, 'label': label, 'round': round_num,
        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'icir': icir, 'ic_mean': ic_mean, '+ic_pct': ic_pct,
        'category': category, 'hypothesis': hypothesis,
        'logic': logic, 'formula': formula,
        'direction': direction, 'status': 'candidate',
    })
    save_pool()


def print_pool():
    """打印因子库"""
    df = get_pool_df()
    print("=" * 70)
    print("  因子库 (XQuant Ch9 通过验证的因子)")
    print("=" * 70)
    print(f"\n  总数: {len(df)}")
    print(f"\n{'名称':20s} {'类别':12s} {'ICIR':>7s} {'+IC%':>7s} {'轮次':>4s} {'状态':>10s}")
    print("-" * 65)
    for _, row in df.iterrows():
        print(f"{row['name']:20s} {row['category']:12s} {row['icir']:+7.3f} "
              f"{row['+ic_pct']:6.1%} {int(row['round']):4d} {row['status']:>10s}")


if __name__ == '__main__':
    df = save_pool()
    print_pool()
    print(f"\n  已保存: {POOL_CSV}")
