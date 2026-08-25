"""Stage 5 QLoRA for Re — first REAL handwritten-paragraph data + fresh doc pages.

Rationale (2026-08-01, eyoun-s4-1 post-mortem): a second epoch of the same mix was a
wash (misraj flat, sedra/historyar gave back ground) — the levers left are NEW data,
not more epochs. This stage introduces:
  * khatt_para: 2,160 REAL KHATT paragraphs (eval-decontaminated) — the khatt suite
    (CER ~4.0, hallucination failure) has never seen real paragraph training data.
  * khatt_lines: ~5k KHATT line crops joining handwritten_line (style match).
  * img2md: 13.7k real Arabic book pages -> markdown (Arabic-Nougat train set),
    fresh doc_markdown mass for the misraj/Baseer gap.

Design (vs stage4_1.py):
  * CONTINUES the eyoun-s4 CHAMPION adapter (eyoun-s4-1 rejected).
  * build_mixed adds SOURCE_WEIGHT overriding DOMAIN_WEIGHT per source: the old
    (saturated) doc pool drops to a 0.25 reminder dose while img2md gets 1.0.
  * handwritten_para_real 3.0 (~6.5k effective) — the khatt lever, small data so
    repeated, hi-res images (built at 1280 vis-token budget).
  * lr 1e-5 (new data, full LR of Stage 4), seed 45.
  * All stability fixes carried (ignore_data_skip, empty_cache EVERY step, hi-res).
Output: checkpoints/stage5, adapter eyoun-s5. Eval: merged eyoun-s2 base + eyoun-s5.
"""
import argparse
import os

BASE = os.environ.get("EYOUN_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA = os.path.join(BASE, "data", "unified")
CKPT_DIR = os.path.join(BASE, "train", "checkpoints", "stage5")
S4_ADAPTER = os.path.join(BASE, "train", "adapters", "eyoun-s4")
HIRES_PIXELS = 1280 * 28 * 28   # must match build_unified_dataset.py

DOMAIN_WEIGHT = {
    "doc_markdown": 0.25,           # old pool = reminder dose (saturated in S4/S4.1)
    "handwritten_para_real": 3.0,   # NEW: real KHATT paragraphs — the khatt lever
    "handwritten_line": 1.0,        # okai + NEW khatt_lines; sedra protection
    "manuscript": 2.0,              # historyar keep-warm
    "line_real": 0.15,
    "newspaper": 0.2,
    "line_synth": 0.05,
    "word_crop": 0.1,
    # handwritten_paragraph (synthetic): still DROPPED (failed domain)
}
SOURCE_WEIGHT = {
    "img2md": 1.0,   # fresh real pages get a FULL dose despite doc_markdown 0.25
}


def build_mixed(ds, seed=45):
    """DOMAIN_WEIGHT with per-source overrides: sources in SOURCE_WEIGHT are pulled
    out first and weighted independently; the rest of their domain follows
    DOMAIN_WEIGHT. <1 subsamples, >1 duplicates — same math as stage1..4."""
    from datasets import concatenate_datasets

    def weighted(sub, w):
        if len(sub) == 0 or w <= 0:
            return None
        if w < 1.0:
            return sub.shuffle(seed=seed).select(range(int(len(sub) * w)))
        reps = int(w)
        frac = w - reps
        chunks = [sub] * reps
        if frac > 0:
            chunks.append(sub.shuffle(seed=seed).select(range(int(len(sub) * frac))))
        return concatenate_datasets(chunks)

    parts = []
    override_sources = set(SOURCE_WEIGHT)
    for src, w in SOURCE_WEIGHT.items():
        sub = ds.filter(lambda ex, s=src: ex["source"] == s, num_proc=4)
        p = weighted(sub, w)
        if p is not None:
            parts.append(p)
    for dom, w in DOMAIN_WEIGHT.items():
        sub = ds.filter(
            lambda ex, d=dom: ex["domain"] == d and ex["source"] not in override_sources,
            num_proc=4)
        p = weighted(sub, w)
        if p is not None:
            parts.append(p)
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
        # EVERY step (Stage-4.1 lesson): hi-res mixes fragment the 12GB pool fast;
        # ~10ms/step vs 17s steps is free insurance.
        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.empty_cache()

    if not os.path.exists(os.path.join(S4_ADAPTER, "adapter_model.safetensors")):
        raise SystemExit(f"missing eyoun-s4 adapter at {S4_ADAPTER}")

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
        seed=45,
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

    out = os.path.join(BASE, "train", "adapters", "eyoun-s5")
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"adapter saved -> {out}")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
