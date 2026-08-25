"""Evaluate a Qwen2.5-VL model (base or +LoRA adapter) on Arabic OCR eval sets.

Usage (inside the training venv):
    python eval/run_eval.py --model $EYOUN_HOME/base_model --suite arocrbench_khattparagraph
    python eval/run_eval.py --model <path> --adapter <lora_dir> --suite all --limit 50

Outputs: eval/results/<run_name>/<suite>.jsonl (per-sample) and summary.json (CER/WER).
Text normalization (applied identically to hyp+ref): NFKC, strip tatweel, strip tashkeel
(diacritics scored separately), unify alef variants, collapse whitespace.
"""
import argparse
import io
import json
import os
import re
import unicodedata

import torch
from PIL import Image

DATASET_DIR = os.environ.get("EYOUN_DATASETS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MAX_PIXELS = 640 * 28 * 28
PROMPT = "استخرج النص من الصورة."

# suite -> (path, kind, image_col, text_col)
SUITES = {
    "arocrbench_arabicocr":      ("ahmedheakl--arocrbench_arabicocr", "hf", "image", "text"),
    "arocrbench_hindawi":        ("ahmedheakl--arocrbench_hindawi", "hf", "image", "text"),
    "arocrbench_historyar":      ("ahmedheakl--arocrbench_historyar", "hf", "image", "text"),
    "arocrbench_khattparagraph": ("ahmedheakl--arocrbench_khattparagraph", "hf", "image", "answer"),
    "arocrbench_patsocr":        ("ahmedheakl--arocrbench_patsocr", "hf", "image", "answer"),
    "arocrbench_isippt":         ("ahmedheakl--arocrbench_isippt", "hf", "image", "text"),
    "arocrbench_synthesizear":   ("ahmedheakl--arocrbench_synthesizear", "hf", "image", "text"),
    "sedra_handwritten":         ("sedra-hugface--arabic-handwritten-ocr-eval", "hf", "image", "text"),
    "misraj_dococr":             ("Misraj-DocOCR", "parquet", "image", "markdown"),
    "nakba_test":                ("Nakba_test", "parquet", "image", "text"),
}

TASHKEEL = re.compile(r"[ً-ْٰـ]")  # harakat + dagger alef + tatweel


def norm(t, strip_tashkeel=True):
    t = unicodedata.normalize("NFKC", t or "")
    if strip_tashkeel:
        t = TASHKEEL.sub("", t)
    t = re.sub(r"[أإآ]", "ا", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def load_suite(name):
    path, kind, img_col, txt_col = SUITES[name]
    full = os.path.join(DATASET_DIR, path)
    rows = []
    if kind == "hf":
        from datasets import load_from_disk
        ds = load_from_disk(full)
        split = ds[list(ds.keys())[0]] if hasattr(ds, "keys") else ds
        for ex in split:
            rows.append((ex[img_col], ex[txt_col]))
    else:
        import pyarrow.parquet as pq
        import glob
        for f in sorted(glob.glob(os.path.join(full, "*.parquet"))):
            for r in pq.read_table(f).to_pylist():
                img = Image.open(io.BytesIO(r[img_col]["bytes"]))
                rows.append((img, r[txt_col]))
    return rows


def load_model(model_path, adapter=None, max_pixels=MAX_PIXELS):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, quantization_config=bnb, device_map="cuda:0", torch_dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
    model.eval()
    return model, processor


@torch.inference_mode()
def ocr(model, processor, image, gen_kwargs=None):
    image = image.convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
    out = model.generate(**inputs, do_sample=False, **(gen_kwargs or {"max_new_tokens": 2048}))
    gen = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--suite", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default=None, help="run name for results folder")
    # Phase-A anti-loop decode (defaults reproduce the original greedy behaviour)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=MAX_PIXELS,
                    help="image downscale cap; raise to match hi-res training (e.g. 1280*28*28)")
    # Output post-processing (inference-time, weights untouched). misraj ground
    # truth uses HTML <table> but the model emits markdown pipe tables — matched
    # format on 0/58 table pages. Opt-in so markdown output stays available.
    ap.add_argument("--md_tables_to_html", action="store_true",
                    help="rewrite markdown pipe tables as HTML <table> in model output")
    args = ap.parse_args()

    post = (lambda t: t)
    if args.md_tables_to_html:
        from postprocess import md_tables_to_html as post
        print("[post] markdown tables -> HTML <table>")

    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    print(f"[decode] {gen_kwargs}")

    import jiwer
    run_name = args.name or (os.path.basename(args.adapter or args.model.rstrip("/\\")))
    out_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)

    model, processor = load_model(args.model, args.adapter, args.max_pixels)
    suites = list(SUITES) if args.suite == "all" else args.suite.split(",")
    summary_path = os.path.join(out_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):  # merge with prior partial runs
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    for suite in suites:
        rows = load_suite(suite)
        if args.limit:
            rows = rows[: args.limit]
        hyps, refs = [], []
        out_path = os.path.join(out_dir, f"{suite}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for i, (img, ref) in enumerate(rows):
                try:
                    hyp = ocr(model, processor, img, gen_kwargs)
                except Exception as e:
                    hyp = ""
                    print(f"  [{suite}#{i}] generation error: {e}")
                hyp = post(hyp)
                f.write(json.dumps({"i": i, "ref": ref, "hyp": hyp}, ensure_ascii=False) + "\n")
                hyps.append(norm(hyp))
                refs.append(norm(ref))
                if (i + 1) % 25 == 0:
                    print(f"  [{suite}] {i+1}/{len(rows)}", flush=True)
        pairs = [(r, h) for r, h in zip(refs, hyps) if r]
        refs_n = [p[0] for p in pairs]
        hyps_n = [p[1] for p in pairs]
        cer = jiwer.cer(refs_n, hyps_n)
        wer = jiwer.wer(refs_n, hyps_n)
        summary[suite] = {"cer": round(cer, 4), "wer": round(wer, 4), "n": len(pairs)}
        # TEDS — structure, which CER/WER cannot see. Only meaningful where the
        # ground truth actually contains tables (misraj: 58/400 pages).
        try:
            from teds import teds_score
            with open(out_path, encoding="utf-8") as fh:
                recs = [json.loads(l) for l in fh]
            ts = [s for s in (teds_score(r["hyp"], r["ref"]) for r in recs) if s is not None]
            if ts:
                summary[suite]["teds"] = round(100 * sum(ts) / len(ts), 2)
                summary[suite]["teds_n"] = len(ts)
        except Exception as e:
            print(f"  [teds] skipped: {type(e).__name__}: {e}")
        t = summary[suite].get("teds")
        print(f"[{suite}] CER={cer:.4f} WER={wer:.4f}"
              + (f" TEDS={t} (n={summary[suite]['teds_n']})" if t is not None else "")
              + f" n={len(pairs)}", flush=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
