"""Build the unified Re training dataset from audited sources.

Each source becomes its own shard under data/unified_shards/<name>, so a crash
never loses completed shards and sources can be (re)built independently:

    python build_unified_dataset.py aallail yousefmd nakba qari10k omar
    python build_unified_dataset.py namaa          # after images downloaded
    python build_unified_dataset.py --merge        # shards -> data/unified + data/val + STATS.md

Schema: {image: Image(), prompt: str, response: str, source: str, domain: str}
Fixes applied: NFKC unicode (Yousefmd presentation forms), HTML->markdown (NAMAA),
image downscale to MAX_PIXELS, empty/corrupt filtering.
"""
import io
import json
import os
import re
import sys
import unicodedata

from datasets import Dataset, Features, Image as HFImage, Value, load_from_disk, concatenate_datasets
from PIL import Image

DATASET_DIR = os.environ.get("EYOUN_DATASETS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"))
OUT_ROOT = os.environ.get("RE_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
SHARD_DIR = os.path.join(OUT_ROOT, "unified_shards")
NAMAA_IMG_DIR = os.path.join(OUT_ROOT, "namaa_images")
MAX_PIXELS = 640 * 28 * 28  # default vision-token budget for 12GB training
# Stage 3 (2026-07-11): misraj eval images are median ~1.5M px but we trained docs
# at 0.5M px -> the model never sees doc detail at test resolution. Give the two
# detail-critical domains a bigger budget; lines/words/word-crops gain nothing from
# more pixels and would only cost VRAM. Rebuild ONLY these shards + re-merge:
#   python build_unified_dataset.py hastyle qari10k omar namaa jayanthmuthu okai_para --force
HIRES_PIXELS = 1280 * 28 * 28   # ~1.0M px, ~1280 vision tokens/page (tunable; watch VRAM)
DOMAIN_MAX_PIXELS = {
    "doc_markdown": HIRES_PIXELS,
    "manuscript": HIRES_PIXELS,
    "handwritten_para_real": HIRES_PIXELS,  # eval decodes at ~1M px; match it
    # everything else -> MAX_PIXELS
}
VAL_PER_DOMAIN = 400        # stratified holdout -> ~1.6k total

FEATURES = Features({
    "image": HFImage(),
    "prompt": Value("string"),
    "response": Value("string"),
    "source": Value("string"),
    "domain": Value("string"),
})

PROMPTS = [
    "استخرج النص من الصورة.",
    "اقرأ محتوى هذه الصورة واكتبه نصاً.",
    "حوّل هذه الصورة إلى نص ماركداون.",
    "Extract the text from this image.",
    "Convert this document image to markdown.",
]


def pick_prompt(i):
    return PROMPTS[i % len(PROMPTS)]


def norm_text(t):
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)          # fixes presentation forms
    t = re.sub(r"[​‎‏﻿]", "", t)  # zero-width/marks
    return t.strip()


def html_to_md(t):
    """NAMAA labels use <p>/<i>/<b>/<u> markup -> markdown."""
    t = re.sub(r"<b>(.*?)</b>", r"**\1**", t, flags=re.S)
    t = re.sub(r"<i>(.*?)</i>", r"*\1*", t, flags=re.S)
    t = re.sub(r"<u>(.*?)</u>", r"\1", t, flags=re.S)
    t = re.sub(r"</p>\s*<p>", "\n\n", t)
    t = re.sub(r"</?p>", "", t)
    return t.strip()


def shrink(img, max_px=MAX_PIXELS):
    img = img.convert("RGB")
    w, h = img.size
    if w * h > max_px:
        s = (max_px / (w * h)) ** 0.5
        img = img.resize((max(28, int(w * s)), max(28, int(h * s))), Image.LANCZOS)
    return img


def to_jpeg_bytes(img, domain=None):
    """Downscale to the domain's pixel budget (hi-res for doc/manuscript)."""
    max_px = DOMAIN_MAX_PIXELS.get(domain, MAX_PIXELS)
    buf = io.BytesIO()
    shrink(img, max_px).save(buf, "JPEG", quality=92)
    return {"bytes": buf.getvalue(), "path": None}


def valid(text, img):
    return bool(text) and len(text) >= 2 and img is not None and min(img.size) >= 12


def save_shard(name, gen):
    out = os.path.join(SHARD_DIR, name)
    ds = Dataset.from_generator(gen, features=FEATURES)
    ds.save_to_disk(out)
    print(f"[{name}] saved {len(ds)} rows -> {out}")


# ---------------- source builders ----------------

def gen_aallail():
    ds = load_from_disk(os.path.join(DATASET_DIR, "aallail--arabic_ocr_synth_2"))["train"]
    for i, ex in enumerate(ds):
        t = norm_text(ex["text"])
        if valid(t, ex["image"]):
            yield {"image": to_jpeg_bytes(ex["image"]), "prompt": pick_prompt(i),
                   "response": t, "source": "aallail_synth2", "domain": "line_synth"}


def gen_yousefmd():
    ds = load_from_disk(os.path.join(DATASET_DIR, "Yousefmd--arabic_ocr_dataset"))["train"]
    for i, ex in enumerate(ds):
        t = norm_text(ex["text"])  # NFKC converts presentation forms to real letters
        if valid(t, ex["image"]):
            yield {"image": to_jpeg_bytes(ex["image"]), "prompt": pick_prompt(i),
                   "response": t, "source": "yousefmd", "domain": "line_real"}


def gen_nakba():
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(DATASET_DIR, "AR_Nakba_data_train__run_Basser_v4_ep_2", "train.parquet"))
    for i, row in enumerate(t.to_pylist()):
        txt = norm_text(row["chosen"])
        img_field = row["images"]
        raw = img_field.get("bytes") if isinstance(img_field, dict) else None
        if not raw or not txt:
            continue
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
        except Exception:
            continue
        if valid(txt, img):
            yield {"image": to_jpeg_bytes(img), "prompt": pick_prompt(i),
                   "response": txt, "source": "nakba_chosen", "domain": "newspaper"}


def gen_qari10k():
    ds = load_from_disk(os.path.join(DATASET_DIR, "melsiddieg--qari-arabic-ocr-10k"))["train"]
    for i, ex in enumerate(ds):
        # two row formats: `messages` (roles '<|User|>'/'<|Assistant|>' with embedded
        # image bytes) OR flat `image` + `assistant_content` columns
        img, txt = None, ""
        try:
            if ex["messages"]:
                msgs = ex["messages"]
                user = next(m for m in msgs if "user" in (m.get("role") or "").lower())
                asst = next(m for m in msgs if "assistant" in (m.get("role") or "").lower())
                raw = user["images"][0]["bytes"]
                img = Image.open(io.BytesIO(raw))
                img.load()
                txt = norm_text(asst.get("content", ""))
            elif ex["image"] is not None:
                img = ex["image"]
                txt = norm_text(ex.get("assistant_content") or "")
        except Exception:
            continue
        if valid(txt, img):
            yield {"image": to_jpeg_bytes(img, "doc_markdown"), "prompt": pick_prompt(i),
                   "response": txt, "source": "qari10k", "domain": "doc_markdown"}


def gen_omar():
    ds = load_from_disk(os.path.join(DATASET_DIR, "Omar-youssef--arabic-ocr-markdown-dataset"))["train"]
    for i, ex in enumerate(ds):
        t = norm_text(ex["markdown"])
        if valid(t, ex["image"]):
            yield {"image": to_jpeg_bytes(ex["image"], "doc_markdown"), "prompt": pick_prompt(i),
                   "response": t, "source": "omar_markdown", "domain": "doc_markdown"}


def gen_namaa():
    ds = load_from_disk(os.path.join(DATASET_DIR, "NAMAA-Space--QariOCR-v0.3-markdown-mixed-dataset"))
    i = 0
    for split in ("train", "validation"):  # keep their test split out entirely
        for ex in ds[split]:
            path = os.path.join(NAMAA_IMG_DIR, ex["image"].rsplit("/", 1)[-1])
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                continue
            try:
                img = Image.open(path)
                img.load()
            except Exception:
                continue
            t = norm_text(html_to_md(ex["text"]))
            if valid(t, img):
                i += 1
                yield {"image": to_jpeg_bytes(img, "doc_markdown"), "prompt": pick_prompt(i),
                       "response": t, "source": "namaa_qari03", "domain": "doc_markdown"}


def gen_jayanthmuthu():
    """Supervisely-format region annotations: each image has `objects`, each
    object optionally carries a tags[].name == 'Transcription' with the text
    for that bounding box. Crop per labeled region -> (crop, text) pairs.
    Domain mapped per source category; unlabeled objects/images are skipped."""
    import pyarrow.parquet as pq
    import glob

    CAT_DOMAIN = {
        "HandwrittenText": "line_real", "Newspaper": "newspaper",
        "Receipt": "line_real", "Invoice": "line_real", "Label": "line_real",
        "BusinessCard": "line_real", "AdminForm": "doc_markdown",
        "OfficialDocument": "doc_markdown", "Book": "doc_markdown",
        "Magazine": "doc_markdown", "Map": "line_real", "Comics": "line_real",
    }
    i = 0
    for f in sorted(glob.glob(os.path.join(
            DATASET_DIR, "JayanthMuthu--arabic-ocr", "data", "*.parquet"))):
        cat = os.path.basename(f).split("-0000")[0]
        domain = CAT_DOMAIN.get(cat, "line_real")
        t = pq.read_table(f)
        for row in t.to_pylist():
            try:
                ann = json.loads(row["annotation"])
                img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
                full_img = Image.open(io.BytesIO(img_bytes))
                full_img.load()
            except Exception:
                continue
            for obj in ann.get("objects", []):
                txt = next((tag.get("value") for tag in obj.get("tags", [])
                            if tag.get("name") == "Transcription" and tag.get("value")), None)
                txt = norm_text(txt) if txt else None
                if not txt:
                    continue
                try:
                    pts = obj["points"]["exterior"]
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    box = (max(0, min(xs)), max(0, min(ys)), min(full_img.width, max(xs)), min(full_img.height, max(ys)))
                    crop = full_img.crop(box)
                except Exception:
                    continue
                if valid(txt, crop):
                    i += 1
                    yield {"image": to_jpeg_bytes(crop, domain), "prompt": pick_prompt(i),
                           "response": txt, "source": f"jayanthmuthu_{cat}", "domain": domain}


IMG_REF = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def gen_hastyle():
    """1,358 historical manuscript pages -> markdown. Labels embed useless
    ![img-N.jpeg](...) refs (no such files at inference) -> stripped.
    Targets the historyar/khatt gap. New domain: manuscript."""
    ds = load_from_disk(os.path.join(DATASET_DIR, "hastyle--arabic-manuscript-ocr"))["train"]
    for i, ex in enumerate(ds):
        t = norm_text(IMG_REF.sub("", ex["text"]))
        t = re.sub(r"\n{3,}", "\n\n", t)
        if valid(t, ex["image"]):
            yield {"image": to_jpeg_bytes(ex["image"], "manuscript"), "prompt": pick_prompt(i),
                   "response": t, "source": "hastyle_manuscript", "domain": "manuscript"}


def gen_okai():
    """TheRealOKAI v3 train split = Muharaf handwritten sentence crops (22,292).
    `script` metadata claims 'printed' but the source (aamijar/muharaf-public)
    is real handwriting -> targets sedra/khatt gap. Rows sourced from
    ahmedheakl/arocrbench_muharaf are benchmark data -> excluded.
    Their validation/test splits are kept out entirely. Domain: handwritten_line."""
    ds = load_from_disk(os.path.join(DATASET_DIR, "TheRealOKAI--arabic-ocr-datasetv3"))["train"]
    for i, ex in enumerate(ds):
        if "arocrbench" in (ex["source"] or ""):
            continue
        t = norm_text(ex["text"])
        if valid(t, ex["image"]):
            yield {"image": to_jpeg_bytes(ex["image"]), "prompt": pick_prompt(i),
                   "response": t, "source": "okai_muharaf", "domain": "handwritten_line"}


def gen_okai_para(lines_per=(3, 6), seed=42):
    """SPECULATIVE (Stage 3, khatt): synthesize handwritten PARAGRAPH images by
    stacking consecutive okai/Muharaf sentence crops vertically, concatenating
    their transcriptions with newlines. khatt is multi-line paragraphs but our only
    handwritten training data is single-sentence crops -> this fabricates the
    missing paragraph structure. Caveat: Phase A showed khatt fails by HALLUCINATION
    (can't read the script), so this may not fix it; treat as an experiment. New
    domain: handwritten_paragraph. RTL text -> stack top-to-bottom (reading order).
    """
    import random
    rng = random.Random(seed)
    ds = load_from_disk(os.path.join(DATASET_DIR, "TheRealOKAI--arabic-ocr-datasetv3"))["train"]
    buf_imgs, buf_txts, out_i = [], [], 0
    target = rng.randint(*lines_per)
    for ex in ds:
        if "arocrbench" in (ex["source"] or ""):
            continue
        t = norm_text(ex["text"])
        if not valid(t, ex["image"]):
            continue
        buf_imgs.append(ex["image"].convert("RGB"))
        buf_txts.append(t)
        if len(buf_imgs) < target:
            continue
        # compose: common width, stack top->bottom with small padding
        W = max(im.width for im in buf_imgs)
        pad = 10
        resized = [im.resize((W, max(1, int(im.height * W / im.width))), Image.LANCZOS)
                   if im.width != W else im for im in buf_imgs]
        H = sum(im.height for im in resized) + pad * (len(resized) + 1)
        canvas = Image.new("RGB", (W + 2 * pad, H), (255, 255, 255))
        y = pad
        for im in resized:
            canvas.paste(im, (pad, y)); y += im.height + pad
        text = "\n".join(buf_txts)
        if valid(text, canvas):
            out_i += 1
            yield {"image": to_jpeg_bytes(canvas, "handwritten_paragraph"),
                   "prompt": pick_prompt(out_i), "response": text,
                   "source": "okai_muharaf_para", "domain": "handwritten_paragraph"}
        buf_imgs, buf_txts = [], []
        target = rng.randint(*lines_per)


def gen_mssqpi():
    """2.16M single-word printed crops -> deterministic 30k subsample (every
    72nd row). Small lexical-coverage dose only; more adds nothing but compute.
    Domain: word_crop."""
    ds = load_from_disk(os.path.join(DATASET_DIR, "mssqpi--Arabic-OCR-Dataset"))["train"]
    for i in range(0, len(ds), 72):
        ex = ds[i]
        t = norm_text(ex["text"])
        if valid(t, ex["image"]):
            yield {"image": to_jpeg_bytes(ex["image"]), "prompt": pick_prompt(i),
                   "response": t, "source": "mssqpi_words", "domain": "word_crop"}


def gen_presightai():
    """presightai/arabic_doc_to_markdown TRAIN split (32,025 pages): REAL scanned
    printed Arabic documents -> markdown with genuine tables/headers — the print-real
    doc data our synthetic doc_markdown shards lack (Stage-4 misraj lever). Labels
    are fenced in a ```markdown code block -> strip the fence. Test split (8k) is
    NOT ingested (kept for optional held-out checks)."""
    import pyarrow.parquet as pq
    import glob
    files = sorted(glob.glob(os.path.join(
        DATASET_DIR, "presightai--arabic_doc_to_markdown", "data", "train-*.parquet")))
    i = 0
    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                raw = (row.get("image") or {}).get("bytes")
                if not raw:
                    continue
                t = norm_text(row.get("markdown") or "")
                # strip the ```markdown fence wrapper
                m = re.match(r"^```(?:markdown)?\s*\n(.*?)\n?```\s*$", t, flags=re.S)
                if m:
                    t = m.group(1).strip()
                try:
                    img = Image.open(io.BytesIO(raw))
                    img.load()
                except Exception:
                    continue
                if valid(t, img):
                    i += 1
                    yield {"image": to_jpeg_bytes(img, "doc_markdown"),
                           "prompt": pick_prompt(i), "response": t,
                           "source": "presightai", "domain": "doc_markdown"}


def _khatt_eval_probes():
    """40-char probes from the arocrbench_khattparagraph reference texts.
    KHATT's 'similar-text' design repeats one passage across many writers, so any
    training paragraph whose text contains a probe could teach the model an eval
    answer verbatim -> excluded regardless of split (measured 2026-07-29:
    1,836/3,996 paragraphs affected, 2,160 clean)."""
    ev = load_from_disk(os.path.join(DATASET_DIR, "ahmedheakl--arocrbench_khattparagraph"))["train"]
    collapse = lambda s: re.sub(r"\s+", " ", norm_text(s)).strip()
    return [collapse(t)[:40] for t in ev["answer"]], collapse


def gen_khatt_para():
    """a-alnaggar/khatt-paragraphs: 3,996 REAL KHATT handwritten paragraph scans +
    transcriptions — the real-paragraph data the khatt suite always lacked (synthetic
    okai_para failed; khatt fails by hallucination). Text files are CP1256, correct
    logical order. Eval-contaminated paragraphs dropped via probe filter.
    New domain: handwritten_para_real (hi-res budget)."""
    import glob
    probes, collapse = _khatt_eval_probes()
    base = os.path.join(DATASET_DIR, "a-alnaggar--khatt-paragraphs")
    i, dropped = 0, 0
    for fp in sorted(glob.glob(os.path.join(base, "proc_images", "*.jpg"))):
        tp = fp.replace("proc_images", "proc_text")[:-4] + ".txt"
        if not os.path.exists(tp):
            continue
        t = norm_text(open(tp, encoding="cp1256", errors="replace").read())
        flat = collapse(t)
        if any(p in flat for p in probes):
            dropped += 1
            continue
        try:
            img = Image.open(fp)
            img.load()
        except Exception:
            continue
        if valid(t, img):
            i += 1
            yield {"image": to_jpeg_bytes(img, "handwritten_para_real"),
                   "prompt": pick_prompt(i), "response": t,
                   "source": "khatt_para", "domain": "handwritten_para_real"}
    print(f"[khatt_para] dropped {dropped} eval-contaminated paragraphs")


def gen_khatt_lines():
    """johnlockejrr/KHATT_v1.0: 6,673 line crops (128px). metadata.csv text is
    CHARACTER-REVERSED (visual LTR order) -> [::-1] restores logical RTL (verified).
    Contamination filtered at the paragraph level: lines grouped by ParaX id, the
    joined text checked against eval probes, whole group dropped on a hit.
    Domain: handwritten_line (joins okai/Muharaf)."""
    import csv
    probes, collapse = _khatt_eval_probes()
    base = os.path.join(DATASET_DIR, "johnlockejrr--KHATT_v1.0_dataset", "data")
    i, dropped = 0, 0
    for split in ("train", "validation", "test"):
        rows = list(csv.DictReader(open(os.path.join(base, split, "metadata.csv"),
                                        encoding="utf-8")))
        paras = {}
        for r in rows:
            pid = r["file_name"].rsplit("_", 1)[0]
            paras.setdefault(pid, []).append(r)
        for pid, lines in paras.items():
            joined = collapse(" ".join(norm_text(r["text"][::-1]) for r in lines))
            if any(p in joined for p in probes):
                dropped += len(lines)
                continue
            for r in lines:
                t = norm_text(r["text"][::-1])
                try:
                    img = Image.open(os.path.join(base, split, r["file_name"]))
                    img.load()
                except Exception:
                    continue
                if valid(t, img):
                    i += 1
                    yield {"image": to_jpeg_bytes(img), "prompt": pick_prompt(i),
                           "response": t, "source": "khatt_lines",
                           "domain": "handwritten_line"}
    print(f"[khatt_lines] dropped {dropped} eval-contaminated lines")


def gen_img2md():
    """MohamedRashad/arabic-img2md TRAIN parquets (13.7k REAL Arabic book pages,
    1400x1867 -> markdown; the Arabic-Nougat train set). Test parquet kept out.
    ![...](...) image refs stripped (hastyle precedent). Domain: doc_markdown."""
    import pyarrow.parquet as pq
    import glob
    i = 0
    for fp in sorted(glob.glob(os.path.join(
            DATASET_DIR, "MohamedRashad--arabic-img2md", "data", "train-*.parquet"))):
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                raw = (row.get("image") or {}).get("bytes")
                if not raw:
                    continue
                t = norm_text(IMG_REF.sub("", row.get("markdown") or ""))
                t = re.sub(r"\n{3,}", "\n\n", t)
                try:
                    img = Image.open(io.BytesIO(raw))
                    img.load()
                except Exception:
                    continue
                if valid(t, img):
                    i += 1
                    yield {"image": to_jpeg_bytes(img, "doc_markdown"),
                           "prompt": pick_prompt(i), "response": t,
                           "source": "img2md", "domain": "doc_markdown"}


BUILDERS = {
    "aallail": gen_aallail,
    "yousefmd": gen_yousefmd,
    "nakba": gen_nakba,
    "qari10k": gen_qari10k,
    "omar": gen_omar,
    "namaa": gen_namaa,
    "jayanthmuthu": gen_jayanthmuthu,
    # ---- Stage 2 sources (2026-07-06). Excluded: HeshamHaroon turath (text-only,
    # no image column), ahmedheakl ar_ocrvqa_instruct (100% English book-cover
    # VQA despite the name), medyas/loay (redundant printed synth).
    "hastyle": gen_hastyle,
    "okai": gen_okai,
    "mssqpi": gen_mssqpi,
    # ---- Stage 3 sources (2026-07-11)
    "okai_para": gen_okai_para,   # synthetic handwritten paragraphs (khatt, speculative)
    # ---- Stage 4 sources (2026-07-19)
    "presightai": gen_presightai,  # 32k REAL printed doc pages -> markdown (misraj lever)
    # ---- Stage 5 sources (2026-08-01)
    "khatt_para": gen_khatt_para,    # REAL KHATT paragraphs, eval-decontaminated (khatt lever)
    "khatt_lines": gen_khatt_lines,  # KHATT line crops, text un-reversed
    "img2md": gen_img2md,            # 13.7k real book pages -> markdown (misraj lever)
}


def merge():
    shards = []
    for name in os.listdir(SHARD_DIR):
        p = os.path.join(SHARD_DIR, name)
        if os.path.isdir(p):
            shards.append(load_from_disk(p))
    full = concatenate_datasets(shards).shuffle(seed=42)

    # stratified holdout
    val_idx, seen = [], {}
    for i, dom in enumerate(full["domain"]):
        if seen.get(dom, 0) < VAL_PER_DOMAIN:
            val_idx.append(i)
            seen[dom] = seen.get(dom, 0) + 1
    val_set = set(val_idx)
    train_idx = [i for i in range(len(full)) if i not in val_set]

    full.select(val_idx).save_to_disk(os.path.join(OUT_ROOT, "val"))
    train = full.select(train_idx)
    train.save_to_disk(os.path.join(OUT_ROOT, "unified"))

    stats = {"total_train": len(train), "total_val": len(val_idx), "by_source": {}, "by_domain": {}}
    for s in train["source"]:
        stats["by_source"][s] = stats["by_source"].get(s, 0) + 1
    for d in train["domain"]:
        stats["by_domain"][d] = stats["by_domain"].get(d, 0) + 1
    with open(os.path.join(OUT_ROOT, "STATS.md"), "w", encoding="utf-8") as f:
        f.write("# Unified dataset stats\n\n```json\n" + json.dumps(stats, indent=2, ensure_ascii=False) + "\n```\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    os.makedirs(SHARD_DIR, exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv   # rebuild even if the shard already exists
    if "--merge" in sys.argv:
        merge()
    else:
        import shutil
        for name in (args or ["yousefmd", "nakba", "qari10k", "omar"]):
            shard_path = os.path.join(SHARD_DIR, name)
            done_marker = os.path.join(shard_path, "dataset_info.json")
            if os.path.exists(done_marker):
                if not force:
                    print(f"[{name}] shard exists, skipping (use --force to rebuild)")
                    continue
                print(f"[{name}] --force: removing existing shard")
                shutil.rmtree(shard_path)
            print(f"[{name}] building...")
            save_shard(name, BUILDERS[name])
