"""Stage 3 QLoRA for Re — targets the two Stage-2 misses: misraj (doc structure)
and khatt (handwritten paragraphs).

Design (differs from stage2.py):
  * BASE = eyoun-s2 MERGED to 16-bit (train/merged/eyoun-s2-merged-16bit), then a FRESH
    LoRA — because Stage 3 UNFREEZES the vision tower (finetune_vision_layers=True),
    which needs new vision LoRA modules eyoun-s2's language-only adapter lacks. Run
    train/merge_eyoun_s2.py first.
  * HIGHER RESOLUTION: doc_markdown + manuscript shards are rebuilt at ~1.0M px
    (build_unified_dataset.py HIRES_PIXELS). misraj eval images are ~1.5M px median
    but Stage 1/2 trained docs at 0.5M -> the model never saw doc detail at test
    resolution. Vision unfreeze + hi-res is the lever for the misraj WER gap.
  * DOMAIN_WEIGHT adds `handwritten_paragraph` (synthetic khatt data) and drops the
    solved domains to reminder doses.
  * lr 1e-5 (below Stage 2's 2e-5 — refinement + vision is sensitive).
  * CARRIES FORWARD the Stage-2 OOM fixes: ignore_data_skip=True (the resume
    data-skip fragments VRAM to the ceiling) + empty_cache callback. VRAM WARNING:
    vision-unfrozen + hi-res is much heavier than Stage 2 (~3.5GB). Watch it; if it
    OOMs, lower HIRES_PIXELS or freeze vision again.
"""
import argparse
import os

BASE = os.environ.get("EYOUN_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA = os.path.join(BASE, "data", "unified")
CKPT_DIR = os.path.join(BASE, "train", "checkpoints", "stage3")
MERGED_BASE = os.path.join(BASE, "train", "merged", "eyoun-s2-merged-16bit")
HIRES_PIXELS = 1280 * 28 * 28   # must match build_unified_dataset.py

# Gap-tuned for the Stage-2 misses. New domain handwritten_paragraph MUST be listed
# or build_mixed drops it. Solved suites (sedra 0.54, printed <0.1) get reminder doses.
DOMAIN_WEIGHT = {
    "doc_markdown": 2.0,            # misraj — now hi-res, push hard
    "manuscript": 4.0,             # historyar/khatt proxy, hi-res (was 8 at low-res)
    "handwritten_paragraph": 3.0,  # khatt — synthetic paragraphs (speculative)
    "handwritten_line": 1.0,       # sedra solved (0.54) — light reminder
    "line_real": 0.3,
    "newspaper": 0.3,
    "line_synth": 0.1,
    "word_crop": 0.2,
}


def build_mixed(ds, seed=42):
    """Apply DOMAIN_WEIGHT: <1 subsamples, >1 duplicates — same as stage1/2."""
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
        # every 3 steps (not 10): hi-res tensors fragment the allocator faster than
        # Stage 2's low-res, and /10 couldn't keep up (crawl at ~130 s/it by step 228).
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 3 == 0:
                torch.cuda.empty_cache()

    if not os.path.exists(os.path.join(MERGED_BASE, "config.json")):
        raise SystemExit(f"missing merged base {MERGED_BASE} — run train/merge_eyoun_s2.py first")

    model, processor = FastVisionModel.from_pretrained(
        MERGED_BASE, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    # Stage 3: UNFREEZE vision. Fresh LoRA on both towers over the eyoun-s2-merged base.
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        random_state=42,
    )
    # let the processor keep doc/manuscript detail up to the shard budget
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
        ignore_data_skip=True,   # carry the Stage-2 fix: the resume data-skip fragments VRAM
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

    out = os.path.join(BASE, "train", "adapters", "eyoun-s3")
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"adapter saved -> {out}")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
