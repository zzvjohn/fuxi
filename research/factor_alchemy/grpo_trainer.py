"""
GRPO Trainer — 基于 Group Relative Policy Optimization 的因子生成 LLM 训练

三层训练管线:
  Phase 1: SFT 预热 (监督微调 — 学习因子公式"语言")

import os
# ── 国内 HF 镜像 + 禁用 Xet ──────────────────────────────
if not os.environ.get('HF_ENDPOINT'):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_XET'] = '0'
  Phase 2: GRPO 在线 RL (K=4 候选 → FRI 奖励 → 优势估计 → PPO-clip → LoRA 更新)
  Phase 3: 评估 (trained vs baseline 因子质量分布对比)

用法:
    python research/factor_alchemy/grpo_trainer.py \\
        --model Qwen/Qwen2.5-4B-Instruct \\
        --sft-data data/grpo_sft_data.json \\
        --output-dir output/grpo_checkpoints \\
        --grpo-rounds 10

要求: CUDA 12.4+, 16GB+ VRAM (4080 足够)
"""

import os
import sys
import json
import copy
import time
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.factor_alchemy.grpo_reward import (
    FastFRIEvaluator,
    build_existing_pool_from_formulas,
)

# ── 常量 ──────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
K_CANDIDATES = 4          # 每组候选数
PPO_EPSILON = 0.2         # PPO clip ε
KL_BETA = 0.01            # KL 惩罚系数
MAX_NEW_TOKENS = 256      # 生成最大 token 数
TEMPERATURE = 0.8         # 生成温度
GRAD_ACCUM_STEPS = 2      # 梯度累积步数
LEARNING_RATE = 1e-4      # LoRA 学习率



# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def parse_factor_from_output(text: str) -> Optional[Dict[str, str]]:
    """从模型输出中解析因子信息。

    期望格式:
        factor_name: xxx
        expression: xxx
        rationale: xxx
    """
    result = {}
    for field in ['factor_name', 'expression', 'rationale']:
        pattern = rf'{field}\s*[:：]\s*(.+?)(?:\n|$)'
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            result[field] = match.group(1).strip()
    return result if 'expression' in result else None


def compute_log_prob(
    model, tokenizer, prompt: str, completion: str, device: str = "cuda"
) -> torch.Tensor:
    """计算 completion 在模型下的对数概率 (仅计算 completion 部分)"""
    full_text = prompt + completion
    full_inputs = tokenizer(full_text, return_tensors="pt", truncation=True,
                            max_length=1024).to(device)

    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=768).to(device)
    prompt_len = prompt_inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model(**full_inputs, labels=full_inputs.input_ids)
        # logits → log probs
        logits = outputs.logits[0, prompt_len-1:-1]  # shift for next-token
        labels = full_inputs.input_ids[0, prompt_len:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

    return token_log_probs.sum()


def compute_group_advantages(
    rewards: List[float], group_size: int = K_CANDIDATES
) -> List[float]:
    """组内相对优势: A_i = (r_i - mean_group) / (std_group + 1e-8)"""
    n_groups = len(rewards) // group_size
    advantages = []

    for g in range(n_groups):
        group_rewards = rewards[g * group_size:(g + 1) * group_size]
        mean_r = np.mean(group_rewards)
        std_r = np.std(group_rewards) + 1e-8
        for r in group_rewards:
            advantages.append((r - mean_r) / std_r)

    return advantages


def estimate_kl_divergence(
    log_probs_new: List[float],
    log_probs_old: List[float],
) -> float:
    """估计 KL(π_new || π_old)"""
    kls = []
    for lp_new, lp_old in zip(log_probs_new, log_probs_old):
        kls.append(lp_old - lp_new)  # log(p_old/p_new) ≈ D_KL
    return np.mean(kls)



# ═══════════════════════════════════════════════════════════
# SFT Dataset
# ═══════════════════════════════════════════════════════════

class FactorSFTDataset(Dataset):
    """因子公式 SFT 数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        for item in raw.get('data', []):
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output = item.get('output', '')

            # 构建 prompt
            if input_text:
                prompt = f"<|im_start|>system\n{instruction}\n上下文: {input_text}<|im_end|>\n<|im_start|>user\n请生成因子公式:<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n请生成因子公式:<|im_end|>\n<|im_start|>assistant\n"

            full_text = prompt + output + "<|im_end|>"

            # Tokenize
            enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            enc['labels'] = enc['input_ids'].copy()

            # Mask prompt tokens (only compute loss on output)
            prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length)
            prompt_len = len(prompt_enc['input_ids'])
            enc['labels'][:prompt_len] = [-100] * prompt_len

            self.samples.append(enc)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



# ═══════════════════════════════════════════════════════════
# GRPO Trainer
# ═══════════════════════════════════════════════════════════

@dataclass
class GRPOTrainingState:
    """GRPO 训练状态追踪"""
    round: int = 0
    mean_reward: float = 0.0
    max_reward: float = 0.0
    mean_advantage: float = 0.0
    kl_divergence: float = 0.0
    loss: float = 0.0
    factors_evaluated: int = 0
    elapsed_seconds: float = 0.0
    top_formulas: List[Dict] = field(default_factory=list)


class GRPOTrainer:
    """GRPO 因子生成训练器"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        device: str = "cuda",
        output_dir: str = "output/grpo_checkpoints",
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[GRPO] Loading model: {model_name}")

        # ── 4-bit 量化配置 ──────────────────────────────
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # ── 加载模型 ────────────────────────────────────
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.use_cache = False  # 训练时需要关闭 KV cache
        
        # 禁用梯度检查点 (Qwen2.5 默认开启, 极度拖慢训练)
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        elif hasattr(self.model.config, 'use_cache'):
            pass  # already disabled

        # ── Tokenizer ───────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── LoRA ────────────────────────────────────────
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # ── 参考模型 (冻结, 用于 KL 估计) ────────────────
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # ── 优化器 ──────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.01,
        )

        # ── 评估器 (lazy init) ──────────────────────────
        self.evaluator: Optional[FastFRIEvaluator] = None

        # ── 训练状态 ────────────────────────────────────
        self.training_history: List[GRPOTrainingState] = []

        # ── 范式 prompt 模板 ────────────────────────────
        self.paradigm_prompts = [
            "生成一个A股流动性类因子公式，使用WorldQuant风格表达式。",
            "生成一个A股动量效应因子公式，使用WorldQuant风格表达式。",
            "生成一个A股短期反转效应因子公式，使用WorldQuant风格表达式。",
            "生成一个A股波动率类因子公式，使用WorldQuant风格表达式。",
            "生成一个A股资金流类因子公式，使用WorldQuant风格表达式。",
            "生成一个A股行为金融类因子公式，使用WorldQuant风格表达式。",
            "生成一个A股微观结构类因子公式，使用WorldQuant风格表达式。",
            "生成一个A股尾部风险类因子公式，使用WorldQuant风格表达式。",
            "生成一个A股趋势跟踪类因子公式，使用WorldQuant风格表达式。",
            "生成一个A股筹码分布类因子公式，使用WorldQuant风格表达式。",
            "生成一个A股基本面代理类因子公式，使用WorldQuant风格表达式。",
            "生成一个A股隔夜效应因子公式，使用WorldQuant风格表达式。",
        ]

        print(f"[GRPO] Model loaded. Trainable params: "
              f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    # ── Phase 1: SFT 预热 ───────────────────────────────

    def sft_warmup(
        self,
        sft_data_path: str,
        epochs: int = 3,
        batch_size: int = 2,
        save_adapter: bool = True,
    ):
        """监督微调预热：在因子池数据上训练基础因子生成能力 (手动训练循环, 避开 accelerate/Trainer 在 Windows 上的 PATH 长度问题)"""
        print(f"\n{'='*60}")
        print(f"[SFT] Phase 1: SFT Warmup ({epochs} epochs)")
        print(f"{'='*60}")

        from torch.utils.data import DataLoader as TorchDataLoader

        dataset = FactorSFTDataset(sft_data_path, self.tokenizer)
        print(f"[SFT] Dataset size: {len(dataset)} samples")

        # 手动 padding collator
        def collate_fn(batch):
            max_len = max(len(item['input_ids']) for item in batch)
            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []
            for item in batch:
                pad_len = max_len - len(item['input_ids'])
                padded_input_ids.append(item['input_ids'] + [self.tokenizer.pad_token_id] * pad_len)
                padded_attention_mask.append(item.get('attention_mask', [1] * len(item['input_ids'])) + [0] * pad_len)
                padded_labels.append(item['labels'] + [-100] * pad_len)
            return {
                'input_ids': torch.tensor(padded_input_ids),
                'attention_mask': torch.tensor(padded_attention_mask),
                'labels': torch.tensor(padded_labels),
            }

        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # 优化器 + 学习率调度
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=2e-4, weight_decay=0.01,
        )
        total_steps = len(dataloader) * epochs
        warmup_steps = max(1, total_steps // 10)

        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(dataloader):
                # Warmup schedule
                if global_step < warmup_steps:
                    lr = 2e-4 * (global_step + 1) / warmup_steps
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr

                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss / GRAD_ACCUM_STEPS
                loss.backward()

                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * GRAD_ACCUM_STEPS
                global_step += 1
                
                if step % 10 == 0:
                    print(f"    Step {step+1}/{len(dataloader)} loss={loss.item()*GRAD_ACCUM_STEPS:.4f}", flush=True)

            avg_loss = epoch_loss / len(dataloader)
            print(f"  [SFT] Epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")

        # 最后一步梯度更新
        if global_step % GRAD_ACCUM_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        if save_adapter:
            adapter_path = self.output_dir / "sft_adapter"
            self.model.save_pretrained(str(adapter_path))
            self.tokenizer.save_pretrained(str(adapter_path))
            print(f"[SFT] Adapter saved to: {adapter_path}")

        # 更新参考模型
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        print(f"[SFT] Warmup complete.\n")

    # ── Phase 2: GRPO 在线训练 ──────────────────────────

    def grpo_train(
        self,
        rounds: int = 10,
        prompts_per_round: int = 3,
        evaluator: Optional[FastFRIEvaluator] = None,
        save_every: int = 3,
    ):
        """GRPO 在线强化学习训练

        Args:
            rounds: 训练轮数
            prompts_per_round: 每轮采样的 prompt 数 (总候选 = ppR × K_CANDIDATES)
            evaluator: FastFRIEvaluator (如 None 则 lazy init)
            save_every: 每 N 轮保存一次 adapter
        """
        self.evaluator = evaluator

        print(f"\n{'='*60}")
        print(f"[GRPO] Phase 2: Online RL ({rounds} rounds, "
              f"{prompts_per_round} prompts/round, K={K_CANDIDATES})")
        print(f"{'='*60}")

        self.model.train()

        for round_idx in range(rounds):
            t0 = time.time()
            state = GRPOTrainingState(round=round_idx + 1)

            # ── 1. 采样 prompts ──────────────────────────
            prompts = self._sample_prompts(prompts_per_round)

            # ── 2. 生成 K 个候选 per prompt ──────────────
            all_completions = []  # [(prompt, completion_text, factor_dict)]
            for prompt in prompts:
                formatted = self._format_prompt(prompt)
                for k in range(K_CANDIDATES):
                    completion_text = self._generate(formatted)
                    factor_dict = parse_factor_from_output(completion_text)
                    all_completions.append((prompt, completion_text, factor_dict))

            # ── 3. 评估奖励 ──────────────────────────────
            rewards = []
            valid_count = 0
            for prompt, completion_text, factor_dict in all_completions:
                if factor_dict and 'expression' in factor_dict:
                    reward = self.evaluator.evaluate(
                        factor_dict['expression'], verbose=False
                    )
                    if reward > 0:
                        valid_count += 1
                        # 追踪 top formulas
                        if reward >= 0.3:
                            state.top_formulas.append({
                                'prompt': prompt[:50],
                                'reward': round(reward, 4),
                                'expression': factor_dict['expression'][:100],
                            })
                else:
                    reward = 0.0
                rewards.append(reward)

            if valid_count == 0:
                print(f"  [Round {round_idx+1}] No valid factors generated, skip")
                continue

            # ── 4. 组内优势 ──────────────────────────────
            advantages = compute_group_advantages(rewards)

            # ── 5. 计算 log probs + PPO loss ────────────
            all_log_probs_new = []
            all_log_probs_old = []
            batch_loss = 0.0
            batch_count = 0

            # 只对有效因子的样本计算 loss
            for i, (prompt, completion_text, factor_dict) in enumerate(all_completions):
                if factor_dict is None or rewards[i] <= 0:
                    continue

                # 格式化
                formatted_prompt = self._format_prompt(prompt)
                formatted_completion = self._format_factor_output(factor_dict)

                # Log probs
                lp_new = compute_log_prob(
                    self.model, self.tokenizer,
                    formatted_prompt, formatted_completion, self.device
                )
                lp_old = compute_log_prob(
                    self.ref_model, self.tokenizer,
                    formatted_prompt, formatted_completion, self.device
                )

                all_log_probs_new.append(lp_new.item())
                all_log_probs_old.append(lp_old.item())

                # PPO-clip loss
                ratio = torch.exp(lp_new - lp_old.detach())
                adv = torch.tensor(advantages[i], device=self.device)
                clipped = torch.clamp(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON)
                ppo_loss = -torch.min(ratio * adv, clipped * adv)
                kl_penalty = KL_BETA * (lp_old.detach() - lp_new)
                loss = (ppo_loss + kl_penalty) / GRAD_ACCUM_STEPS
                loss.backward()
                batch_loss += loss.item()
                batch_count += 1

            # ── 6. 梯度更新 ──────────────────────────────
            if batch_count > 0:
                if (batch_count // GRAD_ACCUM_STEPS) > 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # ── 7. 统计 ──────────────────────────────────
            state.mean_reward = float(np.mean(rewards))
            state.max_reward = float(np.max(rewards))
            state.mean_advantage = float(np.mean([a for a, r in zip(advantages, rewards) if r > 0]))
            state.kl_divergence = estimate_kl_divergence(all_log_probs_new, all_log_probs_old) if all_log_probs_new else 0.0
            state.loss = batch_loss / max(batch_count, 1)
            state.factors_evaluated = len(rewards)
            state.elapsed_seconds = time.time() - t0

            self.training_history.append(state)

            # ── 8. 日志 ──────────────────────────────────
            print(f"  [Round {round_idx+1:2d}/{rounds}] "
                  f"reward={state.mean_reward:.3f} (max={state.max_reward:.3f}) "
                  f"adv={state.mean_advantage:.3f} "
                  f"kl={state.kl_divergence:.4f} "
                  f"loss={state.loss:.4f} "
                  f"valid={valid_count}/{len(rewards)} "
                  f"[{state.elapsed_seconds:.0f}s]")

            # ── 9. 保存 checkpoint ──────────────────────
            if (round_idx + 1) % save_every == 0:
                self._save_checkpoint(round_idx + 1)

        # ── 最终保存 ─────────────────────────────────────
        self._save_checkpoint("final")
        self._save_training_log()

        print(f"\n[GRPO] Training complete. Best reward: {max(s.mean_reward for s in self.training_history):.3f}")
        self._print_best_formulas()

    # ── Phase 3: 评估 ───────────────────────────────────

    def evaluate(
        self,
        baseline_model_name: Optional[str] = None,
        num_prompts: int = 10,
        evaluator: Optional[FastFRIEvaluator] = None,
    ) -> Dict:
        """评估训练后模型 vs 基线的因子质量"""
        evaluator = evaluator or self.evaluator

        print(f"\n{'='*60}")
        print(f"[EVAL] Phase 3: Factor Quality Evaluation")
        print(f"{'='*60}")

        # 使用相同的 prompts 在两个模型上生成
        prompts = self._sample_prompts(num_prompts, seed=99)

        results = {"trained": [], "baseline": []}
        baseline_model = None

        if baseline_model_name:
            print(f"[EVAL] Loading baseline: {baseline_model_name}")
            baseline_model = AutoModelForCausalLM.from_pretrained(
                baseline_model_name,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

        eval_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="left"
        )
        if eval_tokenizer.pad_token is None:
            eval_tokenizer.pad_token = eval_tokenizer.eos_token

        for model_key, model in [("trained", self.model), ("baseline", baseline_model)]:
            if model is None:
                continue

            rewards = []
            for prompt in prompts:
                formatted = self._format_prompt(prompt)
                for k in range(2):  # 2 candidates per prompt
                    if model_key == "trained":
                        completion = self._generate(formatted)
                    else:
                        inputs = eval_tokenizer(formatted, return_tensors="pt",
                                                truncation=True, max_length=768).to(self.device)
                        outputs = baseline_model.generate(
                            **inputs, max_new_tokens=MAX_NEW_TOKENS,
                            temperature=TEMPERATURE, do_sample=True,
                            pad_token_id=eval_tokenizer.eos_token_id,
                        )
                        completion = eval_tokenizer.decode(outputs[0], skip_special_tokens=True)

                    factor_dict = parse_factor_from_output(completion)
                    if factor_dict and 'expression' in factor_dict:
                        reward = evaluator.evaluate(factor_dict['expression'], verbose=False)
                        if reward > 0:
                            rewards.append(reward)
                            results[model_key].append({
                                'prompt': prompt[:50],
                                'reward': round(reward, 4),
                                'expression': factor_dict['expression'][:100],
                            })

            if rewards:
                print(f"  [{model_key:8s}] n={len(rewards)} "
                      f"mean_reward={np.mean(rewards):.3f} "
                      f"max={np.max(rewards):.3f} "
                      f"std={np.std(rewards):.3f}")

        # 保存评估结果
        eval_path = self.output_dir / "evaluation.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results

    # ── 内部方法 ────────────────────────────────────────

    def _sample_prompts(self, n: int, seed: Optional[int] = None) -> List[str]:
        """采样范式 prompt (可重复)"""
        rng = np.random.RandomState(seed) if seed else np.random.RandomState()
        indices = rng.randint(0, len(self.paradigm_prompts), size=n)
        return [self.paradigm_prompts[i] for i in indices]

    def _format_prompt(self, instruction: str) -> str:
        """格式化为 Qwen chat template"""
        return (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n请生成因子公式。输出格式:\n"
            f"factor_name: <名称>\n"
            f"expression: <公式，使用close_p/open_p/high_p/low_p/vol_p>\n"
            f"rationale: <经济学解释，包含维度关键词如动量/流动性/波动率/资金流/尾部风险>\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _format_factor_output(self, factor_dict: Dict[str, str]) -> str:
        """格式化因子为训练输出文本"""
        return (
            f"factor_name: {factor_dict.get('factor_name', '')}\n"
            f"expression: {factor_dict.get('expression', '')}\n"
            f"rationale: {factor_dict.get('rationale', '')[:200]}\n"
            f"<|im_end|>"
        )

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        """生成因子公式"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _save_checkpoint(self, tag):
        """保存 LoRA adapter"""
        path = self.output_dir / f"adapter_round_{tag}"
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        print(f"  [Checkpoint] Saved to {path}")

    def _save_training_log(self):
        """保存训练历史"""
        log_path = self.output_dir / "training_log.json"
        history = []
        for s in self.training_history:
            history.append({
                'round': s.round,
                'mean_reward': s.mean_reward,
                'max_reward': s.max_reward,
                'kl_divergence': s.kl_divergence,
                'loss': s.loss,
                'elapsed_seconds': s.elapsed_seconds,
            })
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"  [Log] Saved to {log_path}")

    def _print_best_formulas(self, top_n: int = 5):
        """打印训练过程中发现的最优因子公式"""
        all_formulas = []
        for state in self.training_history:
            all_formulas.extend(state.top_formulas)
        all_formulas.sort(key=lambda x: x['reward'], reverse=True)

        if all_formulas:
            print(f"\n  Top {min(top_n, len(all_formulas))} formulas discovered during training:")
            for i, f in enumerate(all_formulas[:top_n]):
                print(f"  {i+1}. reward={f['reward']:.3f} | {f['expression'][:80]}")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRPO Factor Generation Trainer')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='Base model name')
    parser.add_argument('--sft-data', default='data/grpo_sft_data.json',
                        help='SFT dataset path')
    parser.add_argument('--output-dir', default='output/grpo_checkpoints',
                        help='Checkpoint output directory')
    parser.add_argument('--sft-epochs', type=int, default=3,
                        help='SFT warmup epochs')
    parser.add_argument('--grpo-rounds', type=int, default=10,
                        help='GRPO training rounds')
    parser.add_argument('--prompts-per-round', type=int, default=3,
                        help='Prompts per round')
    parser.add_argument('--skip-sft', action='store_true',
                        help='Skip SFT warmup')
    parser.add_argument('--skip-grpo', action='store_true',
                        help='Skip GRPO training')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip evaluation')
    args = parser.parse_args()

    model_name = args.model
    sft_data = PROJECT_ROOT / args.sft_data
    output_dir = PROJECT_ROOT / args.output_dir

    print(f"{'='*60}")
    print(f"GRPO Factor Generation Trainer")
    print(f"  Model: {model_name}")
    print(f"  SFT data: {sft_data}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # ── 检查 GPU ────────────────────────────────────────
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available! Please check GPU/nvidia-smi")
        print("   Run: nvidia-smi")
        print("   If 'GPU is lost' → reboot system")
        sys.exit(1)

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 初始化 ──────────────────────────────────────────
    trainer = GRPOTrainer(
        model_name=model_name,
        output_dir=str(output_dir),
    )

    if not args.eval_only:
        # Phase 1: SFT
        if not args.skip_sft:
            if sft_data.exists():
                trainer.sft_warmup(
                    str(sft_data),
                    epochs=args.sft_epochs,
                )
            else:
                print(f"[WARN] SFT data not found: {sft_data}")
                print(f"  Run: python research/factor_alchemy/grpo_data_builder.py --output data/grpo_sft_data.json")

        # Phase 2: GRPO
        if not args.skip_grpo:
            # Lazy init evaluator
            print("[GRPO] Loading market data for reward evaluation...")
            from scripts.factor_injector import load_data
            close_df, open_df, high_df, low_df, vol_df = load_data()

            # 构建已有因子池 (用于 novelty)
            from research.factor_alchemy.fri import build_existing_pool_from_csv
            pool_csv = PROJECT_ROOT / 'data' / 'passed_factor_pool.csv'
            existing_pool = build_existing_pool_from_csv(
                pool_csv, close_df, open_df, high_df, low_df, vol_df
            )
            existing_names = list(existing_pool.keys()) if existing_pool else None

            evaluator = FastFRIEvaluator(
                close_df, open_df, high_df, low_df, vol_df,
                existing_names=existing_names,
            )

            # 将已有因子加入评估器的池
            if existing_pool:
                evaluator._existing_pool = existing_pool
                evaluator.existing_names = set(existing_pool.keys())

            print(f"[GRPO] Evaluator ready. Existing pool: {len(existing_pool) if existing_pool else 0} factors")

            trainer.grpo_train(
                rounds=args.grpo_rounds,
                prompts_per_round=args.prompts_per_round,
                evaluator=evaluator,
            )

        # Phase 3: 评估
        if not args.no_eval and not trainer.training_history:
            print("[EVAL] No training history, skipping evaluation")
        elif not args.no_eval:
            # 需要重新加载 evaluator (可能在不同进程中)
            from scripts.factor_injector import load_data
            close_df, open_df, high_df, low_df, vol_df = load_data()
            evaluator = FastFRIEvaluator(
                close_df, open_df, high_df, low_df, vol_df,
            )
            trainer.evaluate(
                baseline_model_name=model_name,
                evaluator=evaluator,
            )

    elif args.eval_only:
        # 仅评估: 加载已保存的 adapter
        adapter_path = output_dir / "adapter_round_final"
        if adapter_path.exists():
            print(f"[EVAL] Loading trained adapter: {adapter_path}")
            trainer.model = PeftModel.from_pretrained(
                trainer.model, str(adapter_path)
            )
        else:
            print("[EVAL] No trained adapter found, evaluating base model")

        from scripts.factor_injector import load_data
        close_df, open_df, high_df, low_df, vol_df = load_data()
        evaluator = FastFRIEvaluator(
            close_df, open_df, high_df, low_df, vol_df,
        )
        trainer.evaluate(
            baseline_model_name=model_name,
            evaluator=evaluator,
        )

    print(f"\n{'='*60}")
    print("GRPO Training Pipeline Complete!")
    print(f"{'='*60}")
