"""Stage 6.1 — HTML table retargeting at a CORRECTED dose.

Rationale (2026-08-07): first-ever TEDS measurement showed champion eyoun-soup50 at
**TEDS 0.0** on misraj — it emits markdown pipe tables while the benchmark encodes
tables as HTML <table>, so every table page scored a structural zero. A free
output-side converter (eval/postprocess.py) lifts that to 34.3 (37.3 on GT-table
pages) vs Baseer's 66, but 18/58 table pages still score exactly 0 because the
model MIS-SEGMENTS CELLS — markup rewriting cannot fix wrong cell boundaries.
That residual is what this stage trains.

Why HTML targets (and why this is not product lock-in): HTML makes cell boundaries
explicit, so it is the better supervision signal; markdown output remains available
at inference via the deterministic reverse transform.

Data: NO new collection needed — ~17.9k rows in data/unified already carry markdown
tables (presightai is 54% table-bearing, omar_markdown 43%). They are converted to
HTML on the fly here, so no shard rebuild / image re-encode is required.

Design:
  * CONTINUES eyoun-soup50 (the champion, r=32 from the task-vector concat).
  * Stage 6 (TABLE_WEIGHT 2.0) made tables ~64% of the mix when they are ~14.5% of
    real doc pages. It FIXED recognition (TEDS-0 pages 18/58 -> 6/58) but then
    hallucinated tables on 40 clean pages, so net TEDS fell 18.7 -> 15.6 and misraj
    CER worsened 0.387 -> 0.409. Right lever, wrong dose.
  * 6.1 sets TABLE_WEIGHT 0.45 so tables sit near their true frequency (~22% of the
    mix) — keep the recognition gain, drop the over-trigger.
  * lr 8e-6, 1 epoch, seed 48. Hi-res preserved (cell segmentation needs detail).
  * All stability fixes carried (ignore_data_skip, empty_cache EVERY step).
Output: checkpoints/stage6, adapter eyoun-s6. Eval: merged eyoun-s2 base + eyoun-s6, WITH
TEDS reported (eval/teds.py) — CER/WER alone are blind to what this stage targets.
"""
import argparse
import os
import re
import sys

BASE = os.environ.get("EYOUN_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "eval"))
DEFAULT_DATA = os.path.join(BASE, "data", "unified")
CKPT_DIR = os.path.join(BASE, "train", "checkpoints", "stage6_1")
SRC_ADAPTER = os.path.join(BASE, "train", "adapters", "eyoun-soup50")
HIRES_PIXELS = 1280 * 28 * 28

TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)

# Protective mix — everything that is NOT a table row. Doses mirror the ratios that
# held the suites steady in Stage 5, scaled down since tables are the focus here.
DOMAIN_WEIGHT = {
    "doc_markdown": 0.20,           # non-table doc pages: keep the format warm
    "handwritten_para_real": 0.30,
    "handwritten_line": 0.20,
    "manuscript": 0.60,
    "line_real": 0.05,
    "newspaper": 0.08,
    "line_synth": 0.02,
    "word_crop": 0.03,
}
TABLE_WEIGHT = 0.45                 # was 2.0 in Stage 6 -> over-trigger


def has_md_table(t):
    return len(TABLE_ROW.findall(t or "")) >= 2


def build_mixed(ds, seed=49):
    """Table-bearing rows (converted to HTML) upweighted, plus a protective mix."""
    from datasets import concatenate_datasets
    from postprocess import md_tables_to_html

    def weighted(sub, w):
        if len(sub) == 0 or w <= 0:
            return None
        if w < 1.0:
            return sub.shuffle(seed=seed).select(range(int(len(sub) * w)))
        reps, frac = int(w), w - int(w)
        chunks = [sub] * reps
        if frac > 0:
            chunks.append(sub.shuffle(seed=seed).select(range(int(len(sub) * frac))))
        return concatenate_datasets(chunks)

    tables = ds.filter(lambda ex: has_md_table(ex["response"]), num_proc=4)
    tables = tables.map(lambda ex: {"response": md_tables_to_html(ex["response"])},
                        num_proc=4)
    print(f"table rows: {len(tables)} (converted markdown -> HTML)")

    parts = []
    p = weighted(tables, TABLE_WEIGHT)
    if p is not None:
        parts.append(p)
    for dom, w in DOMAIN_WEIGHT.items():
        sub = ds.filter(
            lambda ex, d=dom: ex["domain"] == d and not has_md_table(ex["response"]),
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
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--resume", default="auto", choices=["auto", "no"])
    args = ap.parse_args()

    import torch
    from datasets import load_from_disk
    from unsloth import FastVisionModel, is_bf16_supported
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback

    class EmptyCacheCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.empty_cache()

    if not os.path.exists(os.path.join(SRC_ADAPTER, "adapter_model.safetensors")):
        raise SystemExit(f"missing champion adapter at {SRC_ADAPTER}")

    model, processor = FastVisionModel.from_pretrained(
        SRC_ADAPTER,
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
        learning_rate=8e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=1,
        max_steps=args.max_steps,
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
        seed=49,
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

    out = os.path.join(BASE, "train", "adapters", "eyoun-s6-1")
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"adapter saved -> {out}")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
