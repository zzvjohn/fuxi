"""
GRPO 训练一键启动脚本

用法:
    python scripts/run_grpo.py [--quick] [--full]

    --quick:  快速验证 (SFT 1 epoch + GRPO 3 rounds)
    --full:   完整训练 (SFT 3 epochs + GRPO 10 rounds)
    (默认):   标准训练 (SFT 3 epochs + GRPO 10 rounds)

注意:
    国内需设置 HF_ENDPOINT=https://hf-mirror.com 下载模型
"""

import os
# ── 国内 HuggingFace 镜像 ────────────────────────────────
if not os.environ.get('HF_ENDPOINT'):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_XET'] = '0'  # hf-mirror 不支持 Xet 存储
print(f"[ENV] HF_ENDPOINT → {os.environ['HF_ENDPOINT']}, XET=0")

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='GRPO Training Launcher')
    parser.add_argument('--quick', action='store_true', help='Quick test run')
    parser.add_argument('--full', action='store_true', help='Full training')
    parser.add_argument('--model', default='Qwen/Qwen2.5-Coder-7B-Instruct',
                        help='Base model (default: Qwen2.5-Coder-7B-Instruct)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation on trained adapter')
    parser.add_argument('--skip-sft', action='store_true',
                        help='Skip SFT warmup, go straight to GRPO')
    args = parser.parse_args()

    # ── 步骤 0: 检查 GPU ────────────────────────────────
    print("=" * 60)
    print("Step 0: GPU Check")
    print("=" * 60)

    try:
        import torch
        if not torch.cuda.is_available():
            print("\n❌ CUDA not available!")
            print("Run 'nvidia-smi' to check GPU status.")
            print("If 'GPU is lost' error → reboot system and re-run.")
            sys.exit(1)
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("\n❌ torch not installed. Run:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)

    # ── 步骤 1: 构建 SFT 数据 ────────────────────────────
    sft_data_path = PROJECT_ROOT / 'data' / 'grpo_sft_data.json'
    if not sft_data_path.exists():
        print("\n" + "=" * 60)
        print("Step 1: Build SFT Dataset")
        print("=" * 60)
        result = subprocess.run(
            [sys.executable,
             str(PROJECT_ROOT / 'research/factor_alchemy/grpo_data_builder.py'),
             '--output', str(sft_data_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print("❌ Failed to build SFT dataset")
            print(result.stderr)
            sys.exit(1)
    else:
        print(f"\n✅ SFT data exists: {sft_data_path}")

    # ── 步骤 2: 下载模型 (首次运行, 通过 ModelScope) ──────
    print("\n" + "=" * 60)
    print("Step 2: Model Download (via ModelScope)")
    print("=" * 60)

    model_name = args.model
    # ModelScope 路径: qwen/Qwen2.5-Coder-7B-Instruct
    ms_model_id = model_name.replace('/', '_') if '/' in model_name else model_name
    local_model_dir = PROJECT_ROOT / 'models' / model_name.replace('/', '--')

    if not (local_model_dir / 'config.json').exists():
        print(f"Downloading {model_name} to {local_model_dir}...")
        try:
            from modelscope import snapshot_download
            snapshot_download(
                f'qwen/Qwen2.5-Coder-7B-Instruct',
                cache_dir=str(PROJECT_ROOT / 'models'),
                local_dir=str(local_model_dir),
            )
            print(f"✅ Model downloaded: {local_model_dir}")
        except Exception as e:
            print(f"⚠️  ModelScope download failed: {e}")
            print("   Falling back to HF (may need proxy)...")
    else:
        print(f"✅ Model exists: {local_model_dir}")

    # 使用本地路径
    model_arg = str(local_model_dir) if (local_model_dir / 'config.json').exists() else model_name

    # ── 步骤 3: 训练 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Training")
    print("=" * 60)

    if args.quick:
        sft_flag = ['--skip-sft'] if args.skip_sft else []
        print(f"⚡ Quick mode: {'GRPO only' if args.skip_sft else 'SFT 1 epoch + GRPO 3 rounds'}")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'research/factor_alchemy/grpo_trainer.py'),
            '--model', model_arg,
            '--sft-data', str(sft_data_path),
            '--output-dir', str(PROJECT_ROOT / 'output/grpo_checkpoints'),
            '--sft-epochs', '1',
            '--grpo-rounds', '3',
            '--prompts-per-round', '2',
        ] + sft_flag
    elif args.eval_only:
        print("🔬 Eval-only mode")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'research/factor_alchemy/grpo_trainer.py'),
            '--model', model_arg,
            '--output-dir', str(PROJECT_ROOT / 'output/grpo_checkpoints'),
            '--eval-only',
        ]
    else:
        print("🚀 Standard training: SFT 3 epochs + GRPO 10 rounds")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'research/factor_alchemy/grpo_trainer.py'),
            '--model', model_arg,
            '--sft-data', str(sft_data_path),
            '--output-dir', str(PROJECT_ROOT / 'output/grpo_checkpoints'),
            '--sft-epochs', '3',
            '--grpo-rounds', '10',
            '--prompts-per-round', '3',
        ]

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("\n❌ Training failed!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ GRPO Training Complete!")
    print(f"   Checkpoints: {PROJECT_ROOT / 'output/grpo_checkpoints'}")
    print(f"   Training log: {PROJECT_ROOT / 'output/grpo_checkpoints/training_log.json'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
