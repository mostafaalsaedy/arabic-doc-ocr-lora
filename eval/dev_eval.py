"""Fast dev-slice eval (~130 samples) for sweeping adapters cheaply.

The full 10-suite eval is ~10 h — useless for a lambda sweep. This runs fixed
deterministic slices in CHEAP -> EXPENSIVE order with a collapse canary first:

  arocrbench_arabicocr  20   trivial printed text. eyoun-s4 sits at CER 0.040 and has
                             since Stage 1; eyoun-dpo2 hit 1.782 (44x). Nothing that
                             damages the model leaves this suite intact, so it is
                             the earliest and cheapest collapse detector we have.
  sedra_handwritten     30   handwriting protection
  khattparagraph        40   the hallucination target domain
  misraj_dococr         40   the Baseer benchmark (slowest — runs last)

If the canary blows past the threshold the run ABORTS before spending ~20 min on
misraj. Decode flags mirror the production eval exactly so numbers stay
comparable; slices are rows[:N] (same selection as run_eval --limit).

Usage:
    python scripts/dev_eval.py --adapter train/adapters/eyoun-s4 --name lam0.00
    python scripts/dev_eval.py --adapter <dir> --name lam0.20 --baseline lam0.00
"""
import argparse
import json
import os
import sys
import time

BASE = os.environ.get("EYOUN_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "eval"))

DEV_DIR = os.path.join(BASE, "eval", "results", "dev")
MERGED = os.path.join(BASE, "train", "merged", "eyoun-s2-merged-16bit")

# SLICE SET v2 (2026-08-06). Sweep-1 validation against full-suite actuals showed
# dev reliability tracks SUITE STABILITY, not slice size:
#   arabicocr 0.86 -> 0.865 OK | hindawi 0.89 -> 0.867 OK   (low CER, stable)
#   sedra     0.57 -> 1.00  XX | khatt   1.06 -> 0.956 XX   (high CER, loop-prone:
#     a couple of runaway generations dominate the corpus-level aggregate)
# So sedra/khatt are DROPPED — they cost ~24 min per candidate and their numbers
# were actively misleading. n is unchanged for the retained three so the existing
# lam0.00 (eyoun-s4) baseline stays valid: 0.0402 / 0.1369 / 0.3135.
# MISRAJ STRATIFICATION (2026-08-06) — the fix for a proven blind spot.
# Calibration at λ=1.0 (pure eyoun-s5, whose full-suite misraj is KNOWN to be 1.10x
# worse) had the dev slice reporting a harmless 0.99x. Cause: misraj damage lives
# in a LOOP-FAILURE TAIL, not the median — under eyoun-s4 only 13/400 pages generate
# >1.5x reference length, and ZERO of them fall in the first 40 pages. The slice
# sampled the healthy median (ratio 0.916) and could not see the failure mode.
# Fix: score the fragile tail SEPARATELY from the median instead of averaging the
# tail away. MISRAJ_LOOP = the 24 most loop-prone pages under eyoun-s4 (ratios
# 4.92..1.17); MISRAJ_NORM = 30 well-behaved pages. Report both — "did the median
# hold" and "did the tail blow up" are different questions and only the second
# predicts full-suite misraj regressions.
MISRAJ_LOOP = [32, 35, 47, 63, 79, 81, 108, 138, 158, 179, 198, 210, 216, 221,
               238, 273, 315, 318, 328, 346, 365, 382, 385, 398]
MISRAJ_NORM = list(range(30))

SLICES = [                       # (suite, spec) — spec: int = first N, list = indices
    ("arocrbench_arabicocr", 20),
    ("arocrbench_hindawi", 30),   # eyoun-s5's one genuine win (0.153->0.119) lives here
    ("misraj_dococr", MISRAJ_NORM),   # median behaviour
    ("misraj_dococr", MISRAJ_LOOP),   # THE sensitive detector — watch this one
]
CANARY = "arocrbench_arabicocr"
CANARY_ABS = 0.20      # absolute floor when no baseline is available
CANARY_MULT = 4.0      # or 4x the baseline dev CER, whichever applies

DECODE = {"max_new_tokens": 1024, "repetition_penalty": 1.2, "no_repeat_ngram_size": 6}
MAX_PIXELS = 1003520   # production eval resolution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--baseline", default=None, help="name of a prior dev run to diff against")
    args = ap.parse_args()

    import jiwer
    from run_eval import load_suite, load_model, ocr, norm

    os.makedirs(DEV_DIR, exist_ok=True)
    base = None
    if args.baseline:
        bp = os.path.join(DEV_DIR, f"{args.baseline}.json")
        if os.path.exists(bp):
            base = json.load(open(bp))["suites"]

    print(f"[dev] adapter={args.adapter}  decode={DECODE}", flush=True)
    model, processor = load_model(MERGED, args.adapter, MAX_PIXELS)

    out = {"adapter": args.adapter, "suites": {}, "aborted": False}
    for suite, spec in SLICES:
        full = load_suite(suite)
        if isinstance(spec, int):
            rows, label = full[:spec], suite
        else:                                   # explicit index list (stratified)
            rows = [full[i] for i in spec if i < len(full)]
            label = f"{suite}[{'loop' if spec is MISRAJ_LOOP else 'norm'}]"
        suite_key = label
        t0 = time.time()
        hyps, refs = [], []
        for img, ref in rows:
            try:
                hyp = ocr(model, processor, img, DECODE)
            except Exception as e:
                hyp = ""
                print(f"  [{suite}] gen error: {type(e).__name__}", flush=True)
            hyps.append(norm(hyp))
            refs.append(norm(ref))
        pairs = [(r, h) for r, h in zip(refs, hyps) if r]
        cer = jiwer.cer([p[0] for p in pairs], [p[1] for p in pairs])
        wer = jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs])
        dt = time.time() - t0
        rec = {"cer": round(cer, 4), "wer": round(wer, 4), "n": len(pairs), "sec": round(dt)}
        out["suites"][suite_key] = rec
        delta = ""
        if base and suite_key in base:
            b = base[suite_key]["cer"]
            delta = f"  (baseline {b:.4f}, {cer/b if b else float('inf'):.2f}x)"
        print(f"[dev] {suite_key:28s} CER={cer:.4f} WER={wer:.4f} n={len(pairs)} {dt:.0f}s{delta}",
              flush=True)

        if suite == CANARY:
            limit = base[CANARY]["cer"] * CANARY_MULT if base and CANARY in base else CANARY_ABS
            if cer > limit:
                out["aborted"] = True
                out["abort_reason"] = f"canary CER {cer:.4f} > {limit:.4f}"
                print(f"[dev] ABORT — {out['abort_reason']}; skipping remaining slices",
                      flush=True)
                break

    with open(os.path.join(DEV_DIR, f"{args.name}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[dev] wrote {os.path.join(DEV_DIR, args.name)}.json", flush=True)
    print(json.dumps(out["suites"], indent=2))


if __name__ == "__main__":
    main()
