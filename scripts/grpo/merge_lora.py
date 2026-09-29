import os
import shutil

import torch

BASE = os.environ.get("T1_BASE", "checkpoints/navila-llama3-8b-8f")
RUN = os.environ.get("T1_RUN", "checkpoints/grpo_lora/step_50")
OUT = os.environ.get("T1_MERGED_OUT", "checkpoints/navila-8b-8f-grpo-merged")


def pick_ckpt() -> str:
    if os.path.exists(os.path.join(RUN, "adapter_model.safetensors")):
        return RUN
    cks = sorted(
        (d for d in os.listdir(RUN) if d.startswith("checkpoint-")),
        key=lambda d: int(d.split("-")[1]),
    )
    assert cks, "no checkpoint found under run1"
    return os.path.join(RUN, cks[-1])


def main():
    from peft import PeftModel

    from llava.model import LlavaLlamaModel

    ckpt = pick_ckpt()
    print("merge from:", ckpt, flush=True)

    os.makedirs(OUT, exist_ok=True)
    shutil.copy(os.path.join(BASE, "config.json"), os.path.join(OUT, "config.json"))
    if not os.path.exists(os.path.join(OUT, "vision_tower")):
        shutil.copytree(os.path.join(BASE, "vision_tower"), os.path.join(OUT, "vision_tower"))

    print("loading base model (cpu)...", flush=True)
    model = LlavaLlamaModel.from_pretrained(BASE, low_cpu_mem_usage=True)

    non_lora = os.path.join(ckpt, "non_lora_trainables.bin")
    if os.path.exists(non_lora):
        try:
            w = torch.load(non_lora, map_location="cpu", weights_only=True)
        except Exception:
            w = torch.load(non_lora, map_location="cpu")
        model.load_state_dict(w, strict=False)
        print(f"non-lora trainables applied: {len(w)} tensors", flush=True)
    else:
        print("WARN: non_lora_trainables.bin missing (projector not applied)", flush=True)

    model = PeftModel.from_pretrained(model, ckpt)
    model = model.merge_and_unload()
    print("lora merged", flush=True)

    model = model.to(torch.float16)
    llm_dir = os.path.join(OUT, "llm")
    model.llm.save_pretrained(llm_dir)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        p = os.path.join(BASE, "llm", f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(llm_dir, f))
    print("llm saved", flush=True)

    model.get_mm_projector().save_pretrained(os.path.join(OUT, "mm_projector"))
    print("projector saved", flush=True)
    print("MERGE_DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
