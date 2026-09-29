from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from typing import List, Optional

from scripts.grpo.dataset import prepare_samples
from scripts.grpo.reward import normalize_group, parse_action, reward as reward_fn


def select_batch(samples: List[dict], batch_size: int, rng: random.Random) -> List[dict]:
    if batch_size >= len(samples):
        return list(samples)
    return rng.sample(samples, batch_size)


def batch_action_dist(texts: List[str]) -> dict:
    c = Counter()
    for t in texts:
        a = parse_action(t)
        c[a[0] if a else "unparsed"] += 1
    return dict(c)


def resolve_nav_path(p: str, image_root: Optional[str]) -> Optional[str]:
    from pathlib import Path

    raw = Path(str(p).replace("\\", "/"))
    if raw.is_file():
        return str(raw)
    if raw.is_absolute():
        return str(raw) if os.path.isfile(str(raw)) else None
    if not image_root:
        return None
    relative = raw
    if "scans" in raw.parts:
        idx = raw.parts.index("scans")
        relative = Path(*raw.parts[idx + 1:])
    elif len(raw.parts) > 1:
        relative = Path(*raw.parts[1:])
    joined = str(Path(image_root) / relative)
    return joined if os.path.isfile(joined) else None


class GRPOTrainer:
    def __init__(
        self,
        ckpt: str,
        *,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lr: float = 1e-6,
        beta: float = 0.04,
        clip: float = 0.2,
        G: int = 4,
        temperature: float = 0.9,
        top_p: float = 0.9,
        max_new_tokens: int = 32,
        num_frames: int = 8,
        image_root: Optional[str] = None,
        w_prog: float = 1.0,
    ):
        import torch
        from llava.model.builder import load_pretrained_model

        self.torch = torch
        import torch.distributed as dist

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29567")
            dist.init_process_group(
                backend="nccl", world_size=1, rank=0,
                init_method=f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
            )
        self.ckpt = ckpt
        self.beta = beta
        self.clip = clip
        self.G = G
        self.w_prog = w_prog
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.num_frames = num_frames
        self.image_root = image_root or os.environ.get("NAVCOT_IMAGE_ROOT") or os.environ.get("GRPO_IMAGE_ROOT")

        model_name = os.path.basename(os.path.normpath(ckpt))
        tokenizer, model, image_processor, context_len = load_pretrained_model(ckpt, model_name)
        model = model.cuda()
        self.base_config = model.config
        self.base_device = model.device
        self.base_config.time_token_ids = None
        model = model.to(torch.bfloat16)
        self.dtype = torch.bfloat16

        self._freeze_perception(model)

        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=r".*llm.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        )
        model = get_peft_model(model, lora_cfg)
        self._enable_nonreentrant_grad_checkpoint(model)
        model.print_trainable_parameters()
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        self._assert_perception_frozen()

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr)

        from llava.conversation import SeparatorStyle, conv_templates
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
        from navila_patches.cot_prompt import build_question
        from navila_patches.obs_strategy import sample_and_pad_images

        self._conv_templates = conv_templates
        self._SeparatorStyle = SeparatorStyle
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token
        self._build_question = build_question
        self._sample_and_pad_images = sample_and_pad_images
        self.IGNORE_INDEX = -100

    @staticmethod
    def _freeze_perception(model) -> None:
        for getter in ("get_vision_tower", "get_mm_projector"):
            mod = getattr(model, getter, None)
            if mod is None:
                continue
            sub = mod() if callable(mod) else mod
            if sub is None:
                continue
            for p in sub.parameters():
                p.requires_grad_(False)

    def _assert_perception_frozen(self) -> None:
        vt = getattr(self.model, "get_vision_tower", None)
        vt = vt() if callable(vt) else vt
        if vt is not None:
            n_train = sum(p.numel() for p in vt.parameters() if p.requires_grad)
            assert n_train == 0, f"视觉塔仍有 {n_train} 可训练参数（LoRA 误挂或冻结失败）"

    @staticmethod
    def _enable_nonreentrant_grad_checkpoint(model) -> None:
        import torch.utils.checkpoint as _tc

        if not getattr(_tc, "_grpo_nonreentrant_patched", False):
            _orig = _tc.checkpoint

            def _ckpt_nonreentrant(*args, **kwargs):
                kwargs.setdefault("use_reentrant", False)
                return _orig(*args, **kwargs)

            _tc.checkpoint = _ckpt_nonreentrant
            _tc._grpo_nonreentrant_patched = True

        try:
            model.gradient_checkpointing_enable()
        except Exception:
            model.get_base_model().gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    def _resolve_image_path(self, p: str) -> str:
        resolved = resolve_nav_path(p, self.image_root)
        if resolved is None:
            raise FileNotFoundError(f"无法解析帧路径: {p!r}（image_root={self.image_root!r}）")
        return resolved

    def _load_frames(self, image_paths: List[str]):
        from PIL import Image

        frames = []
        for p in image_paths:
            rp = self._resolve_image_path(p)
            img = Image.open(rp).convert("RGB")
            frames.append(img)
        import numpy as np

        arr = np.asarray(frames[0])
        assert arr.std() > 1e-3, f"首帧疑似全黑/纯色（std={arr.std():.4f}），帧解析可能错误: {image_paths[0]}"
        return frames

    def build_inputs(self, sample: dict):
        torch = self.torch
        frames = self._load_frames(sample["images"])
        frames = self._sample_and_pad_images(frames, num_frames=self.num_frames)

        instruction = sample.get("instruction", "")
        interleaved_images = "<image>\n" * (len(frames) - 1)
        question = self._build_question(instruction=instruction, interleaved_images=interleaved_images)
        memory_text = sample.get("memory_text", "")
        if memory_text:
            question = memory_text + "\n" + question

        conv = self._conv_templates["llama_3"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = self._process_images(frames, self.image_processor, self.base_config).to(
            self.base_device, dtype=self.dtype
        )
        input_ids = (
            self._tokenizer_image_token(prompt, self.tokenizer, self._IMAGE_TOKEN_INDEX, return_tensors="pt")
            .squeeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != self._SeparatorStyle.TWO else conv.sep2
        return input_ids, images_tensor, stop_str

    def generate_candidates(self, input_ids, images_tensor, stop_str):
        torch = self.torch
        texts, comp_ids = [], []
        was_training = self.model.training
        self.model.eval()
        try:
            for _ in range(self.G):
                keywords = [stop_str]
                stopping = self._KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids.unsqueeze(0))
                with torch.inference_mode():
                    out = self.model.generate(
                        input_ids.unsqueeze(0),
                        images=images_tensor,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_new_tokens=self.max_new_tokens,
                        use_cache=True,
                        stopping_criteria=[stopping],
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                new_ids = out[0]
                txt = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                if txt.endswith(stop_str):
                    txt = txt[: -len(stop_str)].strip()
                texts.append(txt)
                comp_ids.append(new_ids)
        finally:
            if was_training:
                self.model.train()
        return texts, comp_ids

    def completion_token_logps(self, input_ids, images_tensor, completion_ids, *, use_ref: bool):
        torch = self.torch
        if completion_ids.numel() == 0:
            return None, None
        merged = torch.cat([input_ids, completion_ids]).unsqueeze(0)
        labels = torch.cat(
            [
                torch.full(
                    (input_ids.shape[0],),
                    self.IGNORE_INDEX,
                    dtype=completion_ids.dtype,
                    device=completion_ids.device,
                ),
                completion_ids,
            ]
        ).unsqueeze(0)

        import contextlib

        if use_ref:
            self.model.eval()
            ctx = self.model.disable_adapter()
        else:
            self.model.train()
            ctx = contextlib.nullcontext()
        with ctx:
            logits, new_labels = self.model.forward(
                input_ids=merged.cuda(),
                images=images_tensor,
                attention_mask=torch.ones_like(merged).cuda(),
                labels=labels.cuda(),
                dpo_forward=True,
            )
        shift_logits = logits[0, :-1, :].float()
        shift_labels = new_labels[0, 1:]
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        valid = shift_labels != self.IGNORE_INDEX
        gather_ids = shift_labels.clamp(min=0).unsqueeze(-1)
        token_logp = log_probs.gather(-1, gather_ids).squeeze(-1)
        mask = valid.float()
        return token_logp, mask

    def _kl_k3(self, policy_logp, ref_logp):
        torch = self.torch
        r = ref_logp - policy_logp
        return torch.exp(r) - r - 1.0

    def grpo_point(self, sample: dict, metrics: dict, grad_scale: float = 1.0):
        torch = self.torch
        input_ids, images_tensor, stop_str = self.build_inputs(sample)
        texts, comp_ids = self.generate_candidates(input_ids, images_tensor, stop_str)

        gt = sample["gt_action"]
        rewards = [
            reward_fn(
                t, gt,
                geo_t=sample.get("geodesic_distance"),
                geo_after=sample.get("geo_after"),
                w_prog=self.w_prog,
            )
            for t in texts
        ]
        advs, degenerate = normalize_group(rewards)
        metrics["gen_texts"].extend(texts)
        metrics["rewards"].extend(rewards)
        if degenerate:
            metrics["degenerate_groups"] += 1
            return None

        valid = [(a, c) for a, c in zip(advs, comp_ids) if c.numel() > 0]
        if not valid:
            return None
        scale = grad_scale / len(valid)
        loss_acc = 0.0
        for adv, cids in valid:
            pol_tok, mask = self.completion_token_logps(input_ids, images_tensor, cids, use_ref=False)
            with torch.no_grad():
                ref_tok, _ = self.completion_token_logps(input_ids, images_tensor, cids, use_ref=True)
            denom = mask.sum().clamp(min=1.0)
            mean_pol = (pol_tok * mask).sum() / denom
            mean_kl = (self._kl_k3(pol_tok, ref_tok) * mask).sum() / denom
            loss_seq = (-adv * mean_pol + self.beta * mean_kl) * scale
            loss_seq.backward()
            loss_acc += float(loss_seq.detach())
            metrics["kl"].append(float(mean_kl.detach()))
        return loss_acc

    def _smoke_probe_logp(self, sample: dict):
        torch = self.torch
        input_ids, images_tensor, stop_str = self.build_inputs(sample)
        texts, comp = self.generate_candidates(input_ids, images_tensor, stop_str)
        cids = comp[0]
        pol_tok, mask = self.completion_token_logps(input_ids, images_tensor, cids, use_ref=False)
        with torch.no_grad():
            ref_tok, _ = self.completion_token_logps(input_ids, images_tensor, cids, use_ref=True)
        assert pol_tok is not None and ref_tok is not None, "completion logp 为空（生成 0 token？）"
        assert pol_tok.shape == ref_tok.shape == mask.shape, (
            f"形状不一致 pol={tuple(pol_tok.shape)} ref={tuple(ref_tok.shape)} mask={tuple(mask.shape)}"
        )
        n = int(mask.sum())
        assert n > 0, "completion mask 全 0（标签扩展/掩码有误）"
        mean_pol = float((pol_tok * mask).sum() / n)
        kl = self._kl_k3(pol_tok, ref_tok)
        mean_kl = float((kl * mask).sum() / n)
        assert math.isfinite(mean_pol) and math.isfinite(mean_kl), f"logp/kl 非有限 pol={mean_pol} kl={mean_kl}"
        print(
            f"[smoke-logp] gen0={texts[0]!r} comp_tokens={n} mean_pol_logp={mean_pol:.3f} "
            f"mean_kl={mean_kl:.5f}（初始应≈0，过大=ref 锚失效）",
            flush=True,
        )

    def train(self, samples: List[dict], *, steps: int, batch_size: int, save_dir: str,
              log_every: int = 5, seed: int = 0, smoke: bool = False, save_every: int = 0):
        torch = self.torch
        rng = random.Random(seed)
        os.makedirs(save_dir, exist_ok=True)
        self.model.train()
        if smoke:
            self._smoke_probe_logp(samples[0])
        for step in range(steps):
            batch = select_batch(samples, batch_size, rng)
            metrics = {"gen_texts": [], "rewards": [], "kl": [], "degenerate_groups": 0}
            self.optimizer.zero_grad()
            grad_scale = 1.0 / max(1, len(batch))
            loss_vals = []
            for s in batch:
                lv = self.grpo_point(s, metrics, grad_scale=grad_scale)
                if lv is not None:
                    loss_vals.append(lv)
            if not loss_vals:
                print(f"[step {step}] 全 batch 退化（零方差），跳过", flush=True)
                continue
            if smoke and step == 0:
                self._smoke_asserts()
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
            self.optimizer.step()

            if step % log_every == 0 or step == steps - 1:
                rw = metrics["rewards"]
                kl = metrics["kl"]
                dist = batch_action_dist(metrics["gen_texts"])
                print(
                    f"[step {step}] loss={sum(loss_vals)/len(loss_vals):.4f} reward_mean="
                    f"{(sum(rw)/len(rw)) if rw else 0:.3f} kl_mean={(sum(kl)/len(kl)) if kl else 0:.5f} "
                    f"degen={metrics['degenerate_groups']} act_dist={dist}",
                    flush=True,
                )
            if save_every and (step + 1) % save_every == 0:
                ck = os.path.join(save_dir, f"step_{step+1}")
                self._save(ck)
                print(f"[save] 中间 ckpt → {ck}", flush=True)
            if smoke and step >= 1:
                print("[smoke] 2 步通过：接口/冻结/logp/帧解析/反向/优化器 OK", flush=True)
                break

        self._save(save_dir)
        print(f"[save] LoRA 权重 → {save_dir}")

    def _smoke_asserts(self):
        torch = self.torch
        vt = getattr(self.model, "get_vision_tower", None)
        vt = vt() if callable(vt) else vt
        if vt is not None:
            bad = [n for n, p in vt.named_parameters() if p.grad is not None]
            assert not bad, f"视觉塔出现梯度（未冻结）: {bad[:3]}"
        tgrads = [p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        assert tgrads, "无可训梯度（LoRA 未接到 loss 或重入 checkpoint 吞梯度）"
        assert all(torch.isfinite(g).all() for g in tgrads), "可训梯度非有限"
        print(f"[smoke-assert] 视觉塔0梯度✓ 可训梯度张量数={len(tgrads)} 全有限✓", flush=True)

    def _save(self, save_dir):
        self.model.save_pretrained(save_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="NaVILA base 权重目录（如 checkpoints/navila-llama3-8b-8f）")
    ap.add_argument("--data", required=True, help="grpo_train.jsonl")
    ap.add_argument("--out", required=True, help="LoRA 输出目录")
    ap.add_argument("--memory-level", default="level1", choices=["off", "level1", "level2"])
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0, help="每 N 步存中间 ckpt（0=只存最终）")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="2 步接口冒烟（1 卡先跑这个）")
    ap.add_argument("--w-prog", type=float, default=1.0, help="测地进度项权重(0=关闭, reward v2.1 隔离实验)")
    args = ap.parse_args()

    samples = prepare_samples(args.data, memory_level=args.memory_level, budget=args.budget)
    print(f"[data] {len(samples)} 训练样本（memory={args.memory_level}）")
    if args.smoke:
        samples = samples[: args.batch_size]

    trainer = GRPOTrainer(
        args.ckpt,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        beta=args.beta,
        G=args.G,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_frames=args.num_frames,
        image_root=args.image_root,
        w_prog=args.w_prog,
    )
    trainer.train(
        samples,
        steps=args.steps,
        batch_size=args.batch_size,
        save_dir=args.out,
        log_every=args.log_every,
        save_every=args.save_every,
        smoke=args.smoke,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
