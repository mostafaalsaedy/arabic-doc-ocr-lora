"""Merge the CHAMPION adapter into the base -> standalone 16-bit Eyoun-3B.

Champion (2026-08-11) = `eyoun-soup61` = 0.5*eyoun-soup50 + 0.5*eyoun-s6-1, an exact LoRA
task-vector soup (r=64, alpha=128, scaling 2.0) over `train/merged/eyoun-s2-merged-16bit`.
This is the Phase-6 packaging artifact and the GGUF conversion source.

    python train/merge_champion.py            # -> train/merged/Eyoun-3B

TWO EARLIER APPROACHES FAILED on this machine — do not retry them:
  1. unsloth `save_pretrained_merged` SILENTLY emitted a copy of the base (output
     safetensors byte-identical, original mtimes preserved). It never applied the
     LoRA. Only caught by comparing tensors — always verify a merge.
  2. PEFT `merge_and_unload()` was OOM-killed (exit 5, no traceback): ~7 GB bf16
     model + merge copies against ~7.5 GB free RAM on this 31 GB box.

So this merges by STREAMING: read the base one shard at a time, add
`scaling * (B @ A)` for any tensor that has a LoRA pair, write the shard out.
Peak memory is one shard (~4 GB) plus the small r=64 adapter. Deltas are computed
in fp32 then cast back to the base dtype.

Key mapping (transformers renamed things between the base and adapter):
    base_model.model.model.language_model.layers.N... -> model.layers.N...
    base_model.model.model.visual....                 -> visual....

NOTE: the champion's deployment config also includes `eval/postprocess.py::
md_tables_to_html`. The merged weights emit MARKDOWN tables; the converter turns them
into the HTML the misraj/Baseer benchmark expects (raw TEDS 2.91 vs 18.34 converted).
Ship the converter with the model.
"""
import json
import os
import shutil

BASE = os.environ.get("EYOUN_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(BASE, "train", "merged", "eyoun-s2-merged-16bit")
ADAPTER = os.path.join(BASE, "train", "adapters", "eyoun-soup61")
OUT = os.path.join(BASE, "train", "merged", "Eyoun-3B")


def base_key(lora_key):
    """LoRA tensor name -> the base weight name it modifies (None if not a LoRA)."""
    if ".lora_A.weight" not in lora_key and ".lora_B.weight" not in lora_key:
        return None
    k = lora_key.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
    k = k.replace("base_model.model.", "", 1)
    if k.startswith("model.language_model."):
        k = "model." + k[len("model.language_model."):]
    elif k.startswith("model.visual."):
        k = k[len("model."):]
    return k + ".weight"


def main():
    if os.path.exists(os.path.join(OUT, "config.json")):
        print(f"merged model already exists -> {OUT} (delete to rebuild)")
        return

    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    cfg = json.load(open(os.path.join(ADAPTER, "adapter_config.json")))
    scaling = cfg["lora_alpha"] / cfg["r"]
    ad = load_file(os.path.join(ADAPTER, "adapter_model.safetensors"))
    print(f"adapter: r={cfg['r']} alpha={cfg['lora_alpha']} scaling={scaling}")

    # group A/B by the base tensor they patch
    pairs = {}
    for k, v in ad.items():
        bk = base_key(k)
        if bk is None:
            continue
        slot = "A" if ".lora_A." in k else "B"
        pairs.setdefault(bk, {})[slot] = v
    pairs = {k: v for k, v in pairs.items() if "A" in v and "B" in v}
    print(f"LoRA pairs to apply: {len(pairs)}")

    index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))
    os.makedirs(OUT, exist_ok=True)

    applied, missing = 0, []
    for shard in shards:
        print(f"  merging {shard} ...", flush=True)
        out = {}
        with safe_open(os.path.join(SRC, shard), "pt") as fh:
            for k in fh.keys():
                t = fh.get_tensor(k)
                if k in pairs:
                    A = pairs[k]["A"].to(torch.float32)
                    B = pairs[k]["B"].to(torch.float32)
                    delta = (B @ A) * scaling
                    if delta.shape != t.shape:
                        missing.append((k, tuple(delta.shape), tuple(t.shape)))
                    else:
                        t = (t.to(torch.float32) + delta).to(t.dtype)
                        applied += 1
                out[k] = t.contiguous()
        save_file(out, os.path.join(OUT, shard), metadata={"format": "pt"})
        del out

    print(f"\napplied {applied}/{len(pairs)} LoRA deltas")
    if missing:
        print(f"SHAPE MISMATCHES ({len(missing)}): {missing[:3]}")
    unapplied = set(pairs) - set(index["weight_map"])
    if unapplied:
        print(f"WARNING: {len(unapplied)} LoRA targets not found in base, e.g. {list(unapplied)[:3]}")

    # copy everything that isn't weights
    for f in os.listdir(SRC):
        if f.endswith(".safetensors"):
            continue
        s = os.path.join(SRC, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(OUT, f))

    # --- verification: a merge that changed nothing is a failure ---
    probe = next(k for k in pairs if k in index["weight_map"])
    with safe_open(os.path.join(SRC, index["weight_map"][probe]), "pt") as fh:
        a = fh.get_tensor(probe).float()
    with safe_open(os.path.join(OUT, index["weight_map"][probe]), "pt") as fh:
        b = fh.get_tensor(probe).float()
    same = torch.equal(a, b)
    print(f"\nverify probe: {probe}")
    print(f"  identical to base : {same}")
    print(f"  max abs diff      : {(a - b).abs().max().item():.6f}")
    if same or applied == 0:
        raise SystemExit("MERGE FAILED — adapter was not applied")
    print(f"\nmerged 16-bit model saved -> {OUT}")


if __name__ == "__main__":
    main()
