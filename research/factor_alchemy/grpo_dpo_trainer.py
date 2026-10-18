"""
DPO Trainer (简化版) — 用已有因子池做离线偏好优化

GPRO 在线 RL 在笔记本 4080 上太慢（每步 30s+ 因子评估），
改用 DPO (Direct Preference Optimization) 离线训练。

原理: 构建 (prompt, 好因子公式, 差因子公式) 三元组,
     训练 LLM 学会偏好高质量因子公式。

用法: python research/factor_alchemy/grpo_dpo_trainer.py
"""

import os, sys, json, time, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 国内镜像 ──────────────────────────────────────────────
if not os.environ.get('HF_ENDPOINT'):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_XET'] = '0'

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader

# ── 常量 ──────────────────────────────────────────────────
BATCH_SIZE = 1          # DPO 需要同时算 chosen + rejected
GRAD_ACCUM = 4
EPOCHS = 5
LEARNING_RATE = 1e-4
DPO_BETA = 0.1          # KL 惩罚系数

# Qwen2.5-Coder-7B prompt 格式
SYSTEM_PROMPT = (
    "你是一个量化因子公式生成器。根据要求生成 WorldQuant 风格的因子表达式。\n"
    "可用变量: close_p, open_p, high_p, low_p, vol_p。\n"
    "可用算子: ts_sum, ts_delta, ts_rank, ts_max, ts_min, ts_mean, pct_change, rolling, rank\n"
    "输出格式:\n"
    "factor_name: <名称>\n"
    "expression: <公式>\n"
    "rationale: <经济学解释>\n"
)


class DPODataset(Dataset):
    """DPO 偏好数据集 — 从因子池提取 (prompt, chosen, rejected) 三元组"""

    def __init__(self, tokenizer, sft_data_path: str, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.triplets = []

        with open(sft_data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        data = raw.get('data', [])
        # 按 quality 分组: high → chosen, low/medium → rejected
        high_samples = [d for d in data if d['metadata'].get('quality') == 'high']
        low_samples = [d for d in data if d['metadata'].get('quality') != 'high']

        print(f"[DPO] High-quality: {len(high_samples)}, Low-quality: {len(low_samples)}")

        # 对每个 high sample, 随机配对一个 low sample 作为 rejected
        rng = np.random.RandomState(42)
        for h_sample in high_samples:
            # 找同范式但低质量的样本
            paradigm = h_sample['metadata'].get('paradigm', '')
            same_paradigm = [s for s in low_samples
                           if s['metadata'].get('paradigm') == paradigm]
            if not same_paradigm:
                same_paradigm = low_samples

            rejected = same_paradigm[rng.randint(0, len(same_paradigm))]

            prompt = SYSTEM_PROMPT + "\n" + h_sample['instruction']
            if h_sample.get('input'):
                prompt += "\n上下文: " + str(h_sample['input'])

            chosen_text = h_sample['output']
            rejected_text = rejected['output']

            self.triplets.append((prompt, chosen_text, rejected_text))

        print(f"[DPO] Triplets: {len(self.triplets)}")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        prompt, chosen, rejected = self.triplets[idx]

        prompt_tokens = self.tokenizer(prompt, truncation=True,
                                        max_length=self.max_length, return_tensors="pt")
        chosen_full = prompt + "\n" + chosen
        rejected_full = prompt + "\n" + rejected

        chosen_enc = self.tokenizer(chosen_full, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
        rejected_enc = self.tokenizer(rejected_full, truncation=True,
                                       max_length=self.max_length, return_tensors="pt")

        prompt_len = prompt_tokens.input_ids.shape[1]
        chosen_len = chosen_enc.input_ids.shape[1]
        rejected_len = rejected_enc.input_ids.shape[1]

        return {
            'prompt_input_ids': prompt_tokens.input_ids[0],
            'prompt_attention_mask': prompt_tokens.attention_mask[0],
            'chosen_input_ids': chosen_enc.input_ids[0],
            'chosen_attention_mask': chosen_enc.attention_mask[0],
            'rejected_input_ids': rejected_enc.input_ids[0],
            'rejected_attention_mask': rejected_enc.attention_mask[0],
            'prompt_len': prompt_len,
            'chosen_len': chosen_len,
            'rejected_len': rejected_len,
        }


def collate_dpo(batch):
    """Pad 所有序列到 batch 最大长度"""
    # 取所有序列的最大长度
    max_p = max(item['prompt_input_ids'].shape[0] for item in batch)
    max_c = max(item['chosen_input_ids'].shape[0] for item in batch)
    max_r = max(item['rejected_input_ids'].shape[0] for item in batch)
    max_len = max(max_p, max_c, max_r)

    pad_id = 151643  # Qwen pad_token_id (eos)

    p_ids, p_mask = [], []
    c_ids, c_mask = [], []
    r_ids, r_mask = [], []
    p_lens, c_lens, r_lens = [], [], []

    for item in batch:
        def pad_to(t, length):
            return torch.cat([t, torch.full((length - t.shape[0],), pad_id, dtype=t.dtype)])

        p_ids.append(pad_to(item['prompt_input_ids'], max_len))
        p_mask.append(pad_to(item['prompt_attention_mask'], max_len))
        c_ids.append(pad_to(item['chosen_input_ids'], max_len))
        c_mask.append(pad_to(item['chosen_attention_mask'], max_len))
        r_ids.append(pad_to(item['rejected_input_ids'], max_len))
        r_mask.append(pad_to(item['rejected_attention_mask'], max_len))

        p_lens.append(item['prompt_len'])
        c_lens.append(item['chosen_len'])
        r_lens.append(item['rejected_len'])

    return {
        'prompt_input_ids': torch.stack(p_ids),
        'prompt_attention_mask': torch.stack(p_mask),
        'chosen_input_ids': torch.stack(c_ids),
        'chosen_attention_mask': torch.stack(c_mask),
        'rejected_input_ids': torch.stack(r_ids),
        'rejected_attention_mask': torch.stack(r_mask),
        'prompt_lens': torch.tensor(p_lens),
        'chosen_lens': torch.tensor(c_lens),
        'rejected_lens': torch.tensor(r_lens),
    }


def dpo_loss(
    model, batch, beta: float = DPO_BETA, device: str = "cuda"
) -> torch.Tensor:
    """DPO loss (单模型版): 用 policy model 算 chosen/rejected log prob, 用相同模型 no_grad 作参考。

    避免双模型 OOM (4080 12GB 不够装两个 7B 模型)。
    """

    # Chosen log prob (trainable)
    c_logits = model(
        input_ids=batch['chosen_input_ids'].to(device),
        attention_mask=batch['chosen_attention_mask'].to(device),
    ).logits

    # Rejected log prob (trainable)
    r_logits = model(
        input_ids=batch['rejected_input_ids'].to(device),
        attention_mask=batch['rejected_attention_mask'].to(device),
    ).logits

    # Reference log prob (no_grad, 用同一模型)
    with torch.no_grad():
        ref_c_logits = model(
            input_ids=batch['chosen_input_ids'].to(device),
            attention_mask=batch['chosen_attention_mask'].to(device),
        ).logits
        ref_r_logits = model(
            input_ids=batch['rejected_input_ids'].to(device),
            attention_mask=batch['rejected_attention_mask'].to(device),
        ).logits

    c_lp = _token_log_prob(c_logits, batch['chosen_input_ids'].to(device),
                           batch['chosen_attention_mask'].to(device),
                           batch['prompt_lens'].to(device))
    r_lp = _token_log_prob(r_logits, batch['rejected_input_ids'].to(device),
                           batch['rejected_attention_mask'].to(device),
                           batch['prompt_lens'].to(device))

    ref_c_lp = _token_log_prob(ref_c_logits, batch['chosen_input_ids'].to(device),
                               batch['chosen_attention_mask'].to(device),
                               batch['prompt_lens'].to(device))
    ref_r_lp = _token_log_prob(ref_r_logits, batch['rejected_input_ids'].to(device),
                               batch['rejected_attention_mask'].to(device),
                               batch['prompt_lens'].to(device))

    c_ratio = c_lp - ref_c_lp.detach()
    r_ratio = r_lp - ref_r_lp.detach()
    logits = beta * (c_ratio - r_ratio)
    loss = -F.logsigmoid(logits).mean()

    return loss


def _token_log_prob(logits, labels, mask, prompt_lens):
    """计算 completion 部分的 token-level log prob (跳过 prompt)"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out prompt tokens
    for i in range(labels.shape[0]):
        token_lp[i, :prompt_lens[i] - 1] = 0.0  # -1 for shift

    # Mask out padding
    shift_mask = mask[..., 1:].contiguous()
    token_lp = token_lp * shift_mask

    return token_lp.sum(dim=-1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='E:/quant/models/Qwen--Qwen2.5-Coder-7B-Instruct')
    parser.add_argument('--sft-data', default='E:/quant/data/grpo_sft_data.json')
    parser.add_argument('--output', default='E:/quant/output/grpo_checkpoints/dpo_adapter')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--quick', action='store_true', help='Quick test with 1 epoch')
    args = parser.parse_args()

    model_path = args.model
    if not Path(model_path).exists():
        # Try HF
        model_path = "Qwen/Qwen2.5-Coder-7B-Instruct"

    print(f"{'='*60}")
    print(f"DPO Factor Preference Trainer")
    print(f"  Model: {model_path}")
    print(f"  Data: {args.sft_data}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── 加载模型 ──────────────────────────────────────────
    print("[DPO] Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # 禁用梯度检查点加速
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA
    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 训练 ──────────────────────────────────────────────
    # 注意: 单模型 DPO — policy 和 reference 是同一个模型,
    # reference log prob 在 no_grad 下计算

    print("[DPO] Building dataset...")
    dataset = DPODataset(tokenizer, args.sft_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_dpo)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=0.01,
    )

    epochs = 1 if args.quick else args.epochs
    print(f"[DPO] Training {epochs} epochs, {len(dataloader)} steps/epoch")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(dataloader):
            loss = dpo_loss(model, batch) / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM

            if step % 5 == 0:
                print(f"  Epoch {epoch+1} Step {step+1}/{len(dataloader)} "
                      f"loss={loss.item()*GRAD_ACCUM:.4f}", flush=True)

        # 末批梯度
        if (len(dataloader)) % GRAD_ACCUM != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / len(dataloader)
        print(f"  Epoch {epoch+1} avg_loss={avg_loss:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # ── 保存 ──────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"[DPO] Adapter saved to {output_path}")

    # ── 快速评估 ──────────────────────────────────────────
    print(f"\n[DPO] Quick evaluation — generating sample factors:")
    model.eval()
    test_prompts = [
        "生成一个A股流动性类因子公式",
        "生成一个A股尾部风险类因子公式",
        "生成一个A股趋势跟踪类因子公式",
    ]

    for prompt in test_prompts:
        formatted = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                          max_length=384).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, temperature=0.7,
                do_sample=True, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 只显示 assistant 部分
        if "assistant\n" in result:
            result = result.split("assistant\n", 1)[1]
        print(f"\n--- {prompt} ---")
        print(result[:300])

    print(f"\n{'='*60}")
    print("DPO Training Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
