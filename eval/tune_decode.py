"""Fast decode-config sweep for misraj loopers (Phase A tuning).

Picks known LOOPER pages (default hyp/ref>2) and known GOOD pages (0.8-1.2) from
the existing eyoun-s2 greedy run, regenerates each under several decode configs, and
reports mean CER + mean length-ratio per group. Goal: a config that fixes loopers
WITHOUT harming good pages. ~12 samples x N configs, a few minutes.
"""
import os, sys, json
import torch, jiwer

BASE = os.environ.get("EYOUN_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "eval"))
import run_eval as E  # reuse load_model / load_suite / ocr / PROMPT

DEFAULT = os.path.join(BASE, "eval", "results", "eyoun-s2", "misraj_dococr.jsonl")
CONFIGS = [
    ("greedy (baseline)", {"max_new_tokens": 2048}),
    ("rep1.2",            {"max_new_tokens": 2048, "repetition_penalty": 1.2}),
    ("rep1.3",            {"max_new_tokens": 2048, "repetition_penalty": 1.3}),
    ("rep1.2+nr6",        {"max_new_tokens": 2048, "repetition_penalty": 1.2, "no_repeat_ngram_size": 6}),
    ("rep1.15+nr8",       {"max_new_tokens": 2048, "repetition_penalty": 1.15, "no_repeat_ngram_size": 8}),
]
N_PER_GROUP = 6


def cer(ref, hyp):
    try:
        return jiwer.cer(ref, hyp)
    except Exception:
        return float("nan")


def main():
    rows = [json.loads(l) for l in open(DEFAULT, encoding="utf-8")]
    ratios = [(i, len(r["hyp"]) / max(len(r["ref"]), 1)) for i, r in enumerate(rows)]
    loopers = [i for i, x in ratios if x > 2][:N_PER_GROUP]
    good = [i for i, x in ratios if 0.8 <= x <= 1.2][:N_PER_GROUP]
    print(f"looper idxs: {loopers}")
    print(f"good   idxs: {good}")

    suite_rows = E.load_suite("misraj_dococr")
    model, processor = E.load_model(os.path.join(BASE, "base_model"),
                                    os.path.join(BASE, "train", "adapters", "eyoun-s2"))

    # cache images+refs for the chosen indices
    picks = {i: (suite_rows[i][0], suite_rows[i][1]) for i in loopers + good}

    print(f"\n{'config':<18} {'loop_CER':>9} {'loop_len':>9} {'good_CER':>9} {'good_len':>9}")
    for name, gk in CONFIGS:
        res = {}
        for grp, idxs in (("loop", loopers), ("good", good)):
            cers, lens = [], []
            for i in idxs:
                img, ref = picks[i]
                hyp = E.ocr(model, processor, img, gk)
                cers.append(cer(ref, hyp))
                lens.append(len(hyp) / max(len(ref), 1))
            res[grp] = (sum(cers) / len(cers), sum(lens) / len(lens))
        print(f"{name:<18} {res['loop'][0]:>9.3f} {res['loop'][1]:>9.2f} {res['good'][0]:>9.3f} {res['good'][1]:>9.2f}", flush=True)


if __name__ == "__main__":
    main()
