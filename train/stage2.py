"""Stage 2 QLoRA training for Re — continues from the Stage-1 adapter (eyoun-s1).

Prereq: rebuild the unified set so it includes the Stage-2 shards
(hastyle/okai/mssqpi/jayanthmuthu):
    python scripts/build_unified_dataset.py --merge

Usage (inside venv):
    python train/stage2.py                    # full run (resumes automatically)
    python train/stage2.py --data <path>      # override dataset

Differences vs stage1.py:
  * model = base + eyoun-s1 adapter, training CONTINUES on the same LoRA modules
    (decoder-only; vision stays frozen — images were pre-resized to MAX_PIXELS
    at shard-build time, so raising resolution would need a shard rebuild).
  * lower LR (2e-5) — refinement pass, not fresh adaptation.
  * DOMAIN_WEIGHT covers the 3 NEW domains (manuscript / handwritten_line /
    word_crop). build_mixed DROPS unlisted domains — keep this dict complete.
  * TODO(gap-driven): after eyoun-s1 eval lands, raise weights for the domains
    matching the worst suites (manuscript->historyar/khatt,
    handwritten_line->sedra/khatt, doc_markdown->misraj) before launching.
"""
import argparse
import os

MAX_PIXELS = 640 * 28 * 28
BASE = os.environ.get("EYOUN_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA = os.path.join(BASE, "data", "unified")
CKPT_DIR = os.path.join(BASE, "train", "checkpoints", "stage2")
S1_ADAPTER = os.path.join(BASE, "train", "adapters", "eyoun-s1")

# steps-share per domain. Gap-tuned 2026-07-07 from eyoun-s1 eval:
# khatt 4.43 (REGRESSED, 200/200 loop >2x ref) / historyar 1.69 / misraj CER
# 1.01 WER 1.57 vs Baseer 0.25 (73/400 loop) — vs printed/synth solved
# (<=0.11) and hindawi 0.19 / nakba 0.35 acceptable.
DOMAIN_WEIGHT = {
    "line_synth": 0.15,       # solved (patsocr .026/synthesizear .073); reminder dose
    "line_real": 0.5,         # hindawi 0.19 — acceptable, halve
    "doc_markdown": 1.5,      # misraj = the Baseer bar; extra half-epoch + jayanthmuthu receipts/invoices
    "newspaper": 0.5,         # nakba 0.348 — acceptable, halve
    "manuscript": 8.0,        # historyar/khatt proxy; 8 epochs of 1,356 pages, more risks overfit
    "handwritten_line": 3.0,  # khatt is the worst suite; muharaf crops are the direct data
    "word_crop": 0.3,         # lexical dose; single words don't address paragraph looping
}


def build_mixed(ds, seed=42):
    """Apply DOMAIN_WEIGHT: <1 subsamples, >1 duplicates (int + frac) — same
    logic as stage1.py."""
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

    # Periodically release cached blocks so allocator fragmentation from Stage-2's
    # high image-size variance can't ratchet reserved VRAM toward the 12GB ceiling
    # over the run (see the ignore_data_skip note below for the full diagnosis).
    class EmptyCacheCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 10 == 0:
                torch.cuda.empty_cache()

    # Loading the ADAPTER DIR makes unsloth attach eyoun-s1's LoRA to the base
    # model and keep training those same modules — no get_peft_model here.
    model, processor = FastVisionModel.from_pretrained(
        S1_ADAPTER,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_training(model)

    ds = load_from_disk(args.data)
    ds = build_mixed(ds)
    print(f"training rows (after mix): {len(ds)}")

    cfg = SFTConfig(
        output_dir=CKPT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
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
        # ROOT-CAUSE FIX (2026-07-09): the OOM/crawl is caching-allocator
        # FRAGMENTATION from Stage-2's high image-size variance (word-crops ->
        # full pages) ratcheting *reserved* VRAM to the 12GB ceiling. Stage-1's
        # uniform line images never triggered it (peaked 4.18GB). On RESUME it's
        # lethal: HF Trainer replays/re-collates the 160k already-seen samples to
        # restore data position, fragmenting the pool to the ceiling BEFORE step
        # 10000 starts -> instant crawl. Isolated tests: fresh=5.6GB flat,
        # resume+ignore_data_skip=5.6GB flat, resume+skip -> 11.9GB crawl. Skipping
        # the replay is the fix; minor data-order drift on a 1-epoch shuffled
        # refinement is acceptable. (Per-sample peak is only ~3.2GB; max_length and
        # a vision-token cap were both proven irrelevant and reverted/removed.)
        ignore_data_skip=True,
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

    out = os.path.join(BASE, "train", "adapters", "eyoun-s2")
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"adapter saved -> {out}")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
