"""Stage 4.1 QLoRA for Re — second epoch on the real-print doc mix.

Rationale (2026-07-22, eyoun-s4 post-mortem): Stage 4's presightai lever delivered the
biggest single-stage misraj gain (WER 0.642->0.560, CER 0.500->0.391) and train loss
ended at 0.88 — clearly UNSATURATED on the new real-print domain (cf. 0.35 on the
synthetic-heavy Stage-3.1 mix). Cheapest proven continuation: one more epoch of the
same mix, continuing from eyoun-s4.

Design (vs stage4.py):
  * CONTINUES the eyoun-s4 adapter (same LoRA modules; base resolves to
    train/merged/eyoun-s2-merged-16bit via adapter_config).
  * SAME DOMAIN_WEIGHT (worked: misraj jumped, sedra held within 0.05, rest flat).
  * lr 5e-6 (half of Stage 4 — second epoch over seen data, avoid overfitting
    the synthetic remainder while presightai keeps learning).
  * seed 43 for the mix shuffle -> different sample order/subsamples than epoch 1.
  * All stability fixes carried (ignore_data_skip, empty_cache/3, hi-res budget).
Output: checkpoints/stage4_1, adapter eyoun-s4-1. Eval: merged eyoun-s2 base + eyoun-s4-1.
"""
import argparse
import os

BASE = os.environ.get("EYOUN_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA = os.path.join(BASE, "data", "unified")
CKPT_DIR = os.path.join(BASE, "train", "checkpoints", "stage4_1")
S4_ADAPTER = os.path.join(BASE, "train", "adapters", "eyoun-s4")
HIRES_PIXELS = 1280 * 28 * 28   # must match build_unified_dataset.py

# Same mix as Stage 4 (proven): push real-print docs, protect solved domains.
DOMAIN_WEIGHT = {
    "doc_markdown": 1.0,       # 79k rows, 40% real print — still the lever
    "handwritten_line": 1.0,   # sedra protection (0.463 after S4; keep the dose)
    "manuscript": 2.0,         # historyar keep-warm (956 rows)
    "line_real": 0.2,
    "newspaper": 0.2,
    "line_synth": 0.05,
    "word_crop": 0.1,
    # handwritten_paragraph: DROPPED (failed domain)
}


def build_mixed(ds, seed=43):   # seed 43: fresh order/subsamples vs Stage 4's 42
    """Apply DOMAIN_WEIGHT: <1 subsamples, >1 duplicates — same as stage1/2/3/4."""
    from datasets import concatenate_datasets
    parts = []
    for dom, w in DOMAIN_WEIGHT.items():
        sub = ds.filter(lambda ex, d=dom: ex["domain"] == d, num_proc=4)
        if len(sub) == 0:
            continue
        if w < 1.0:
            sub = sub.shuffle(seed=seed).select(range(int(len(sub) * w)))
        elif w > 1.0:
            reps = int(w)
            frac = w - reps
            chunks = [sub] * reps
            if frac > 0:
                chunks.append(sub.shuffle(seed=seed).select(range(int(len(sub) * frac))))
            sub = concatenate_datasets(chunks)
        parts.append(sub)
    return concatenate_datasets(parts).shuffle(seed=seed)


def make_collator(model, processor):
    from unsloth.trainer import UnslothVisionDataCollator
    inner = UnslothVisionDataCollator(model, processor)

    def collate(examples):
        convs = []
        for ex in examples:
            convs.append({"messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": ex["image"].convert("RGB")},
                    {"type": "text", "text": ex["prompt"]},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": ex["response"]}]},
            ]})
        return inner(convs)

    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--resume", default="auto", choices=["auto", "no"])
    args = ap.parse_args()

    import torch
    from datasets import load_from_disk
    from unsloth import FastVisionModel, is_bf16_supported
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback

    class EmptyCacheCallback(TrainerCallback):
        # EVERY step (2026-07-26): this run crawled twice at ~11.9GB even with /3
        # (steps 2537, 2837) — seed-43 ordering hits worse image-size variance than
        # Stage 4. empty_cache costs ~10ms vs 17s steps; per-step is the safe max.
        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.empty_cache()

    if not os.path.exists(os.path.join(S4_ADAPTER, "adapter_model.safetensors")):
        raise SystemExit(f"missing eyoun-s4 adapter at {S4_ADAPTER}")

    # Adapter-dir load: continue eyoun-s4's LoRA modules (stage2.py pattern).
    model, processor = FastVisionModel.from_pretrained(
        S4_ADAPTER,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_training(model)
    processor.image_processor.size.longest_edge = HIRES_PIXELS

    ds = load_from_disk(args.data)
    ds = build_mixed(ds)
    print(f"training rows (after mix): {len(ds)}")

    cfg = SFTConfig(
        output_dir=CKPT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=1,
        optim="paged_adamw_8bit",
        bf16=is_bf16_supported(),
        fp16=not is_bf16_supported(),
        logging_steps=10,
        save_steps=400,
        save_total_limit=4,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        max_length=4096,
        seed=43,
        ignore_data_skip=True,   # resume data-skip fragments VRAM (Stage-2 fix)
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor.tokenizer,
        data_collator=make_collator(model, processor),
        train_dataset=ds,
        args=cfg,
        callbacks=[EmptyCacheCallback()],
    )

    resume = None
    if args.resume == "auto" and os.path.isdir(CKPT_DIR):
        ckpts = [d for d in os.listdir(CKPT_DIR) if d.startswith("checkpoint-")]
        if ckpts:
            resume = os.path.join(CKPT_DIR, max(ckpts, key=lambda c: int(c.split("-")[1])))
            print(f"resuming from {resume}")

    stats = trainer.train(resume_from_checkpoint=resume)
    print(stats)

    out = os.path.join(BASE, "train", "adapters", "eyoun-s4-1")
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"adapter saved -> {out}")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
