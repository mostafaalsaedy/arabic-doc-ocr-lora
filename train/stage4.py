"""Stage 4 QLoRA for Re — real-print doc refinement (the misraj lever, take 2).

Rationale (2026-07-19 misraj decomposition of eyoun-s3b, CER 0.50/WER 0.64 vs Baseer
0.25): the residual gap is BROAD text error on real printed docs (stripped CER
0.376 across 257/400 mid-band pages), not a loop tail (23 pages) nor tables. Our
doc_markdown pool was ~100% synthetic renders; the eval is real scanned print.
Fix: presightai/arabic_doc_to_markdown — 32k REAL printed Arabic pages -> markdown
(tables, mixed AR/EN, Arabic-Indic numerals) ingested at hi-res (Stage-4 shard,
2026-07-19). doc_markdown is now 79k rows, 40% real.

Design:
  * CONTINUES the eyoun-s3b adapter (adapter-dir load, stage2.py pattern — eyoun-s3b's
    LoRA already spans vision+language from the Stage-3 fresh-LoRA setup; its
    base resolves to train/merged/eyoun-s2-merged-16bit via adapter_config).
  * DOMAIN_WEIGHT: doc_markdown 1.0 (one full epoch of the enlarged, 40%-real pool
    — presightai is the new signal); handwritten_line 1.0 reminder (sedra 0.414 is
    the best ever — protect it, Stage-3 lesson: never starve a solved domain);
    manuscript 2.0 reminder; the rest trace doses. handwritten_paragraph stays
    dead. Mix ~118k rows ~7.4k steps ~33h.
  * lr 1e-5 cosine (refinement), hi-res processor budget as Stage 3.
  * CARRIES the stability fixes: ignore_data_skip=True + empty_cache EVERY 3 STEPS
    (hi-res shards fragment the allocator; /10 is insufficient — Stage-3 lesson).
Output: checkpoints/stage4, adapter eyoun-s4. Eval loads merged eyoun-s2 base + eyoun-s4.
"""
import argparse
import os

BASE = os.environ.get("EYOUN_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA = os.path.join(BASE, "data", "unified")
CKPT_DIR = os.path.join(BASE, "train", "checkpoints", "stage4")
S3B_ADAPTER = os.path.join(BASE, "train", "adapters", "eyoun-s3b")
HIRES_PIXELS = 1280 * 28 * 28   # must match build_unified_dataset.py

# Stage-4 mix: push the enlarged real-print doc pool, protect solved domains.
DOMAIN_WEIGHT = {
    "doc_markdown": 1.0,       # 79k rows, 40% real print (presightai) — the lever
    "handwritten_line": 1.0,   # sedra 0.414 solved — protective reminder dose
    "manuscript": 2.0,         # historyar 0.342 — small domain (956), keep warm
    "line_real": 0.2,
    "newspaper": 0.2,
    "line_synth": 0.05,
    "word_crop": 0.1,
    # handwritten_paragraph: DROPPED (failed domain, Stage-3 post-mortem)
}


def build_mixed(ds, seed=42):
    """Apply DOMAIN_WEIGHT: <1 subsamples, >1 duplicates — same as stage1/2/3."""
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
        # every 3 steps: hi-res tensors fragment the allocator faster than low-res
        # (Stage-3 lesson; /10 allowed a crawl to ~130 s/it).
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 3 == 0:
                torch.cuda.empty_cache()

    if not os.path.exists(os.path.join(S3B_ADAPTER, "adapter_model.safetensors")):
        raise SystemExit(f"missing eyoun-s3b adapter at {S3B_ADAPTER}")

    # Adapter-dir load attaches eyoun-s3b's LoRA (vision+language) and keeps training
    # those same modules — no get_peft_model (stage2.py pattern).
    model, processor = FastVisionModel.from_pretrained(
        S3B_ADAPTER,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_training(model)
    # keep doc/manuscript detail up to the shard budget (matches Stage 3/3.1)
    processor.image_processor.size.longest_edge = HIRES_PIXELS

    ds = load_from_disk(args.data)
    ds = build_mixed(ds)
    print(f"training rows (after mix): {len(ds)}")

    cfg = SFTConfig(
        output_dir=CKPT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
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
        seed=42,
        ignore_data_skip=True,   # Stage-2 fix: resume data-skip fragments VRAM
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

    out = os.path.join(BASE, "train", "adapters", "eyoun-s4")
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"adapter saved -> {out}")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
