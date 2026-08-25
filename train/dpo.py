"""DPO anti-hallucination stage for Re — preference pairs over faithful vs drifted OCR.

Rationale (2026-08-03, eyoun-s5 post-mortem): Stages 4.1 + 5 proved more SFT data is a
dead lever — the remaining failures are BEHAVIORAL: under uncertainty the model
drifts into hallucination/looping (khatt 4.0 CER, misraj loop-pages that ground
35-50 min against no_repeat_ngram during eval). DPO trains the *preference* for
faithful transcription over plausible drift — the mechanism SFT cannot touch.

Data: nakaba_data_for_dpo_negative_from_muhraf_only/train.parquet — 15,962
(image, chosen, rejected) triplets; rejected = hallucinated/drifted transcription
of the same newspaper crop. Subsampled to N_PAIRS (domain skew: all newspaper;
keep the dose moderate, beta conservative — goal is behavior, not domain).

Design:
  * Policy = eyoun-s4 CHAMPION adapter (4-bit QLoRA, unsloth). ref_model=None ->
    TRL disables the adapter for reference logprobs (no second model in VRAM).
  * beta 0.1, lr 5e-6 cosine (LoRA-DPO range), 1 epoch over the subsample.
  * Images at the SFT newspaper budget (640*28*28) — DPO runs 4 forwards/step;
    hi-res would blow the 12GB ceiling for zero gain on behavior.
  * All stability fixes carried (ignore_data_skip, empty_cache EVERY step).
Output: checkpoints/dpo, adapter eyoun-dpo. Eval: merged eyoun-s2 base + eyoun-dpo.
"""
import argparse
import io
import os

from datasets import disable_caching
disable_caching()  # dataset.map() disk-caches process_row by content fingerprint,
# which collided across script edits (stale prompt_input_ids at an old image
# resolution survived a code change) — force every row transform to run fresh.

BASE = os.environ.get("EYOUN_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = os.environ.get("EYOUN_DATASETS", os.path.join(BASE, "datasets"))
NAKBA_PAIRS = os.path.join(DATASETS, "nakaba_data_for_dpo_negative_from_muhraf_only", "train.parquet")
ONPOLICY_DIR = os.path.join(BASE, "data", "onpolicy_pairs")
CKPT_DIR = os.path.join(BASE, "train", "checkpoints", "dpo2")
S4_ADAPTER = os.path.join(BASE, "train", "adapters", "eyoun-s4")
MAX_PIXELS = 448 * 28 * 28   # 351232. DPO does 4 forwards/step (chosen+rejected x
# policy+ref) vs SFT's 1 — hi-res 1003520 (1290 img tokens) OOM'd at step 2 (578s/it);
# 501760 (v2 first attempt) OOM'd at step 3 too (allocator fragmentation ratchets up
# each step, same mechanism as resume-fragmentation-fix, faster here from 4x the
# alloc/free churn per step). Pairs GENERATED at eval resolution (gen_onpolicy_pairs
# .py); load_pairs resizes down for training only — signal survives, eval runs full-res.
N_PAIRS = 8000
SEED = 46

PROMPT = "استخرج النص من الصورة."


def load_pairs(limit=N_PAIRS, seed=SEED, source="onpolicy"):
    """Parquet(s) -> TRL conversational DPO rows: prompt/chosen/rejected + images.
    source 'onpolicy' (v2 default): chunked parquets from gen_onpolicy_pairs.py.
    source 'nakba': the v1 off-policy set (kept for reference — it collapsed decode)."""
    import glob
    import random
    import pyarrow.parquet as pq
    from PIL import Image

    if source == "onpolicy":
        files = sorted(glob.glob(os.path.join(ONPOLICY_DIR, "chunk_*.parquet")))
        if not files:
            raise SystemExit(f"no on-policy chunks in {ONPOLICY_DIR} — run gen_onpolicy_pairs.py")
        tables = [b for f in files for b in pq.ParquetFile(f).iter_batches(batch_size=256)]
    else:
        tables = list(pq.ParquetFile(NAKBA_PAIRS).iter_batches(batch_size=256))

    all_rows = [r for b in tables for r in b.to_pylist()]
    random.Random(seed).shuffle(all_rows)
    all_rows = all_rows[:limit]

    rows = []
    if True:
        for r in all_rows:
            raw = (r.get("images") or {}).get("bytes")
            ch, rj = (r.get("chosen") or "").strip(), (r.get("rejected") or "").strip()
            if not raw or not ch or not rj or ch == rj:
                continue
            try:
                img = Image.open(io.BytesIO(raw))
                img.load()
                img = img.convert("RGB")
            except Exception:
                continue
            w, h = img.size
            if w * h > MAX_PIXELS:
                s = (MAX_PIXELS / (w * h)) ** 0.5
                img = img.resize((max(28, int(w * s)), max(28, int(h * s))))
            rows.append({
                "images": [img],
                "prompt": [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ]}],
                "chosen": [{"role": "assistant", "content": [{"type": "text", "text": ch}]}],
                "rejected": [{"role": "assistant", "content": [{"type": "text", "text": rj}]}],
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=N_PAIRS)
    ap.add_argument("--source", default="onpolicy", choices=["onpolicy", "nakba"])
    ap.add_argument("--max_steps", type=int, default=-1)  # smoke: e.g. 10
    ap.add_argument("--resume", default="auto", choices=["auto", "no"])
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from unsloth import FastVisionModel, is_bf16_supported
    from trl import DPOTrainer, DPOConfig
    from trl.trainer.dpo_trainer import DataCollatorForPreference
    from transformers import TrainerCallback

    # ---- Qwen2.5-VL plumbing for TRL DPO --------------------------------------
    # TRL 0.24 forwards pixel_values/pixel_attention_mask/image_sizes but NOT
    # Qwen's image_grid_thw (smoke test: rot_pos_emb crashes on grid_thw=None).
    # Three light hooks, valid ONLY at per_device_train_batch_size=1 (no padding,
    # so reshaping stacked pixel_values back to Qwen's flat 2D layout is exact):
    #   1. process_row keeps image_grid_thw in the tokenized row
    #   2. the collator batches it and stashes it in a side channel
    #   3. a model.forward wrapper injects the grid (doubled for TRL's
    #      chosen+rejected concat) and flattens pixel_values to (patches, dim)
    _grid_box = {}

    class QwenPrefCollator(DataCollatorForPreference):
        def torch_call(self, examples):
            grids = [ex.pop("image_grid_thw", None) for ex in examples]
            out = super().torch_call(examples)
            if grids[0] is not None:
                # keep the grid IN the batch — a global side-channel races with
                # accelerate's dataloader prefetch (grid N+1 vs pixels N)
                out["image_grid_thw"] = torch.as_tensor(grids).reshape(-1, 3)
            elif not _grid_box.get("warned"):
                print("[collator] WARNING: image_grid_thw missing from rows", flush=True)
                _grid_box["warned"] = True
            return out

    class _GridProxy:
        """Wraps the model only for the duration of concatenated_forward: injects
        this batch's image_grid_thw (doubled for TRL's chosen+rejected concat) and
        flattens pixel_values from TRL's stacked (2B, npatch, dim) to Qwen's 2D."""
        def __init__(self, m, grid):
            object.__setattr__(self, "_m", m)
            object.__setattr__(self, "_grid", grid)

        def __call__(self, input_ids, **kw):
            # prompt-only length (83) vs prompt+completion input_ids (121) —
            # the model rebuilds it from input_ids when absent
            kw.pop("mm_token_type_ids", None)
            pv = kw.get("pixel_values")
            if pv is not None and kw.get("image_grid_thw") is None and self._grid is not None:
                g = self._grid.to(pv.device)
                if pv.dim() == 3:
                    reps = pv.shape[0] // g.shape[0]
                    kw["image_grid_thw"] = g.repeat(reps, 1)
                    kw["pixel_values"] = pv.reshape(-1, pv.shape[-1])
                else:
                    kw["image_grid_thw"] = g
                if not _grid_box.get("announced"):
                    exp_tok = (kw["image_grid_thw"].prod(dim=-1).sum() // 4).item()
                    n_img_tok = (input_ids == 151655).sum().item()  # Qwen2.5-VL <|image_pad|>
                    tag = "OK" if exp_tok == n_img_tok else "MISMATCH"
                    print(f"[grid-inject] pixel_values {tuple(kw['pixel_values'].shape)}, "
                          f"grid {kw['image_grid_thw'].tolist()}, expected_img_feat={exp_tok}, "
                          f"n_image_pad_in_ids={n_img_tok} [{tag}]", flush=True)
                    _grid_box["announced"] = True
            return self._m(input_ids, **kw)

        def __getattr__(self, k):
            return getattr(object.__getattribute__(self, "_m"), k)

    class QwenDPOTrainer(DPOTrainer):
        def concatenated_forward(self, model, batch, is_ref_model=False):
            grid = batch.get("image_grid_thw")
            return super().concatenated_forward(_GridProxy(model, grid), batch, is_ref_model)

    def patch_vision_process_row():
        """The unsloth-compiled trainer maps a MODULE-LEVEL row fn (not
        self.process_row) that drops image_grid_thw — wrap it in sys.modules."""
        import sys
        mods = [m for n, m in sys.modules.items()
                if "UnslothDPOTrainer" in n and hasattr(m, "dpo_trainer_vision_process_row")]
        if not mods:
            raise SystemExit("could not find compiled UnslothDPOTrainer module to patch")
        for m in mods:
            orig = m.dpo_trainer_vision_process_row

            def patched(features, processing_class, *a, _orig=orig, **kw):
                out = _orig(features, processing_class, *a, **kw)
                proc = processing_class(images=features.get("images"),
                                        text=features.get("prompt", ""),
                                        add_special_tokens=False)
                if "image_grid_thw" in proc:
                    out["image_grid_thw"] = proc["image_grid_thw"][0]
                    # Qwen pixel_values are (npatches, dim) — [0] in the original
                    # fn keeps only the first PATCH ROW; restore the full tensor.
                    out["pixel_values"] = proc["pixel_values"]
                return out

            m.dpo_trainer_vision_process_row = patched
        print(f"[patch] dpo_trainer_vision_process_row patched in {len(mods)} module(s)")

    # ---------------------------------------------------------------------------

    class EmptyCacheCallback(TrainerCallback):
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

    rows = load_pairs(args.pairs, source=args.source)
    print(f"DPO pairs: {len(rows)} (source={args.source})")
    ds = Dataset.from_list(rows)

    cfg = DPOConfig(
        output_dir=CKPT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # v2: 235 pairs -> ~30 steps/epoch (was ~15)
        learning_rate=3e-6,        # v2: lower — v1 at 5e-6 collapsed decode
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=3,        # v2: small on-policy set needs more passes
        max_steps=args.max_steps,
        beta=0.05,                 # v2: half of v1 — bound the logp squeeze
        rpo_alpha=1.0,             # v2: SFT anchor on chosen — hold faithful logps UP
        precompute_ref_log_probs=True,  # v2b: ref forward runs ONCE upfront (small
        # 235-pair set) instead of every step -> halves per-step compute/memory
        # (only policy chosen+rejected forwards remain in the training loop)
        optim="paged_adamw_8bit",
        bf16=is_bf16_supported(),
        fp16=not is_bf16_supported(),
        logging_steps=5,
        save_steps=30,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        max_prompt_length=None,   # never truncate through vision tokens
        max_completion_length=320,  # v2b: trimmed from 512 — quadratic attn cost
        # DPOConfig max_length DEFAULTS TO 1024 — a full-hires image alone produced
        # up to 1290 image_pad tokens; the default truncates INTO the image-token
        # block (text side shrinks, but pixel_values/grid stay full) -> exact
        # tokens/features mismatch we hit at 1003520px. At 351232px images run
        # ~450 img tokens + ~30 text + up to 320 completion -> 1024 has margin.
        max_length=1024,
        seed=SEED,
        ignore_data_skip=True,    # resume data-skip fragments VRAM (Stage-2 fix)
    )

    assert cfg.per_device_train_batch_size == 1, "grid plumbing requires batch=1"
    patch_vision_process_row()
    trainer = QwenDPOTrainer(
        model=model,
        ref_model=None,           # PEFT: adapter-disabled forward = reference
        args=cfg,
        train_dataset=ds,
        processing_class=processor,
        data_collator=QwenPrefCollator(pad_token_id=processor.tokenizer.pad_token_id),
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

    out = os.path.join(BASE, "train", "adapters", "eyoun-dpo2")
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"adapter saved -> {out}")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
