# RetainVLN
A simple VLN work for anti-forgetting, whose baseline is [NaVILA](https://github.com/AnjieCheng/NaVILA).

## 📁 Package Layout

The directory tree mirrors the NaVILA repository: merge `evaluation/` and `scripts/` into your NaVILA root and everything is importable in place.

```graphql
├─ evaluation
|   ├─ navila_patches          # monkey-patch overlay (sits beside vlnce_baselines)
|   |    ├─ apply_patch.py     # monkey_patch() entry: prompt/frames/parsing injection
|   |    ├─ m4_measure.py      # M4_POSE measure: pose + geodesic distance into infos
|   |    ├─ obs_strategy.py    # history-frame sampling (default = original NaVILA behavior)
|   |    ├─ cot_prompt.py      # build_question / output parsing (NAVI_COT=0 = baseline prompt)
|   |    ├─ trace.py           # NAVI_TRACE=1: per-step trajectory + action entropy logging
|   |    └─ perturb.py         # perturbation injector (default off)
|   ├─ eval_metrics.py         # SR / SPL / OSR / NE from per-episode result json
|   └─ scripts
|        └─ run_eval_grpo_279.sh   # 8-GPU sharded evaluation chain (edit paths first)
├─ scripts
|   ├─ grpo                    # ⭐ KL-GRPO: data builder, dataset, reward, trainer, LoRA merge
|   ├─ m4_memory               # M4a memory: events / region grid / manager / text rendering
|   └─ t3_terminal
|        └─ render_terminal.py # data-lineage dependency (path densification used by the builder)
```

## 📊 Evaluation

### Installation

1. Set up the NaVILA evaluation environment (`navila-eval`, Habitat-Sim/Lab v0.1.7, VLN-CE) exactly as described in the **Evaluation → Installation** section of the [NaVILA README](https://github.com/NVlabs/NaVILA#-evaluation).
2. Download the R2R VLN-CE annotations and MP3D scene data into `evaluation/data` (same layout as NaVILA).
3. Download the base checkpoint [a8cheng/navila-llama3-8b-8f](https://huggingface.co/a8cheng/navila-llama3-8b-8f).
4. Merge this overlay into the NaVILA root and export the path convention (used everywhere below):

```bash
cp -r evaluation scripts $NAVILA/           # $NAVILA = your NaVILA repo root
export PYTHONPATH=$NAVILA/evaluation:$NAVILA
```

### The frozen 279-episode benchmark

All comparisons in this project use `val_unseen_5scenes_300.json.gz` (279 episodes over 5 scenes), a frozen subset of R2R val_unseen. **Do not** mix results with the full `val_unseen.json.gz` (1839 episodes / 11 scenes) — the two are different protocols.

```
data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_5scenes_300.json.gz
```

### Running evaluation

All configurations share one command template; only the checkpoint and the `NAVI_MEMORY` switch change. From `$NAVILA/evaluation`:

```bash
NAVI_MEMORY=level1 NAVI_MEMORY_BUDGET=400 NAVI_COT=0 python -c "
from navila_patches.apply_patch import monkey_patch
monkey_patch()
from run import main
main()
" --run-type eval --exp-config vlnce_baselines/config/r2r_baselines/navila.yaml \
    --num-chunks 1 --chunk-idx 0 EVAL_CKPT_PATH_DIR $CKPT \
    TASK_CONFIG.DATASET.DATA_PATH data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_5scenes_300.json.gz \
    RESULTS_DIR eval_out/
```

* **M0 baseline** — `EVAL_CKPT_PATH_DIR=$NAVILA_BASE_CKPT` with `NAVI_MEMORY=off`.
* **M4a memory(aka LTS in the paper)** — same base checkpoint with `NAVI_MEMORY=level1 NAVI_MEMORY_BUDGET=400`.
* **KL-GRPO** — merged checkpoint (see below) with `NAVI_MEMORY=level1 NAVI_MEMORY_BUDGET=400` (the validated 53.4 recipe).

On a multi-GPU machine, use the sharded chain instead (8 chunks, automatic retry for missing shards; edit the paths at the top of the script first):

```bash
bash evaluation/scripts/run_eval_grpo_279.sh
```

### Getting the KL-GRPO checkpoint

Training produces a LoRA adapter (`step_50/` directory). Merge it into the base weights on CPU (~25 s) to obtain a checkpoint the standard NaVILA loader accepts:

```bash
T1_RUN=/path/to/adapter/step_50 \
T1_BASE=/path/to/navila-llama3-8b-8f \
T1_MERGED_OUT=/path/to/navila-8b-8f-grpo-merged \
python scripts/grpo/merge_lora.py
```

Our own LoRA will be soon updated at HuggingFace.

(The `T1_*` environment variable names are historical; the script merges any LLM-only LoRA adapter of this project.)

### Metrics

Evaluation writes one json per shard (`val_unseen_{num_chunks}-{chunk_idx}.json`). Merge the shards, then compute SR/SPL/OSR/NE:

```bash
python -c "
import json, glob
m = {}
for f in sorted(glob.glob('eval_out/**/val_unseen_*.json', recursive=True)):
    m.update(json.load(open(f, encoding='utf-8-sig')))
json.dump(m, open('merged_279.json', 'w'))
"
python evaluation/eval_metrics.py merged_279.json --compare baseline_279.json
```

### Environment variables

| Variable | Values | Default | Meaning |
|---|---|---|---|
| `NAVI_MEMORY` | `off` / `level1` / `level2` | `off` | M4a language-memory injection level |
| `NAVI_MEMORY_BUDGET` | int | `400` | memory context token budget |
| `NAVI_COT` | `0` / `1` | `0` | zero-shot CoT prompt (validated recipe: off) |
| `NAVI_TRACE` | `0` / `1` | `0` | per-step trajectory + first-token entropy logging |
| `NAVI_OBS_MODE` | `linspace` / ... | `linspace` | frame sampling; keep default (= original NaVILA) |

## 🚀 Training (KL-GRPO)

1. Build the training set (R2R/RxR clean points + rendered terminal points, each row carrying prefix poses for offline memory reconstruction):

```bash
python -m scripts.grpo.build_grpo_v2 --src <terminal_mix.jsonl> --gt <r2r_train.json.gz> --out-dir data/grpo_v2
```

2. Run the validated recipe (single GPU, ~24 GB; β=0.12, checkpoint every 50 steps, evaluate `step_50`):

```bash
python -m scripts.grpo.trainer \
  --ckpt $NAVILA_BASE_CKPT \
  --data data/grpo_v2/grpo_train_v2.jsonl \
  --out checkpoints/grpo_run --image-root <mp3d_skybox_scans_root> \
  --memory-level level1 --budget 400 \
  --steps 100 --batch-size 4 --G 6 --temperature 1.0 --top-p 0.95 \
  --beta 0.12 --lr 1e-6 --lora-r 16 --lora-alpha 32 \
  --save-every 50 --log-every 2 --seed 0
```

Reward: pure discrete ground-truth action matching (v1). Progress-based and stop-shaped reward variants were ablated and are strictly worse under the clean protocol; `--w-prog 0` disables the progress term.

## 📜 Acknowledgement

Built on [NaVILA](https://github.com/NVlabs/NaVILA) (RSS'25) and [VLN-CE](https://github.com/jacobkrantz/VLN-CE).
