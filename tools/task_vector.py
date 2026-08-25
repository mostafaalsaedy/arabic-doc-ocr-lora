"""Build an EXACT linear combination of LoRA adapters (task-vector arithmetic).

Motivation (2026-08-05, post eyoun-dpo2 collapse): eyoun-dpo2 and eyoun-s5 were both trained
FROM eyoun-s4, so `delta_i = W_i - W_s4` are task vectors over one shared base. A
rejected adapter can still contain recoverable signal buried under drift damage
(eyoun-s5 genuinely won hindawi 0.153->0.119); interpolating lets us dial the delta
down instead of taking it all-or-nothing.

    W(c) = W_base + scaling * SUM_i c_i * (B_i @ A_i)      with SUM_i c_i = 1

EXACTNESS: naive elementwise blending of A and B is WRONG — (B1+B2)@(A1+A2) has
cross terms B1@A2 + B2@A1 that do not belong. Instead concatenate along the rank
axis, which realises the sum of products exactly:
    A' = [A_1 ; A_2 ; ...]           (sum_r, in)
    B' = [c_1*B_1 , c_2*B_2 , ...]   (out, sum_r)   =>  B'@A' = SUM_i c_i B_i@A_i
Rank grows to sum(r_i), so lora_alpha is scaled by the same factor to keep
peft's scaling = lora_alpha / r invariant. Valid for any c_i (incl. <0 or >1).

Usage:
    python scripts/task_vector.py --out train/adapters/mix-l020 \
        --mix eyoun-s4:0.80 --mix eyoun-dpo2:0.20
    (coefficients must sum to 1.0; c=0 entries are dropped)
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

BASE = os.environ.get("EYOUN_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTER_DIR = os.path.join(BASE, "train", "adapters")


def resolve(name):
    return name if os.path.isdir(name) else os.path.join(ADAPTER_DIR, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", action="append", required=True,
                    help="adapter:coefficient, repeatable (coeffs must sum to 1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    parts = []
    for spec in args.mix:
        name, _, coeff = spec.rpartition(":")
        parts.append((resolve(name), float(coeff)))
    total = sum(c for _, c in parts)
    if abs(total - 1.0) > args.tol:
        raise SystemExit(f"coefficients must sum to 1.0, got {total}")
    parts = [(p, c) for p, c in parts if abs(c) > args.tol]
    if not parts:
        raise SystemExit("all coefficients are zero")

    cfgs, states = [], []
    for path, _ in parts:
        cfgs.append(json.load(open(os.path.join(path, "adapter_config.json"))))
        states.append(load_file(os.path.join(path, "adapter_model.safetensors")))

    ref = cfgs[0]
    for c, (path, _) in zip(cfgs, parts):
        for k in ("target_modules", "use_rslora", "base_model_name_or_path"):
            if c.get(k) != ref.get(k):
                raise SystemExit(f"adapter mismatch on '{k}': {path}")
    keys = set(states[0])
    for s, (path, _) in zip(states, parts):
        if set(s) != keys:
            raise SystemExit(f"tensor-key mismatch: {path}")

    out_state = {}
    for k in sorted(keys):
        if ".lora_A." in k:                      # (r, in)  -> concat on dim 0
            out_state[k] = torch.cat([s[k].to(torch.float32) for s in states], 0)
        elif ".lora_B." in k:                    # (out, r) -> concat on dim 1, scaled
            out_state[k] = torch.cat(
                [s[k].to(torch.float32) * c for s, (_, c) in zip(states, parts)], 1)
        else:
            raise SystemExit(f"unexpected non-LoRA tensor {k} — concat trick invalid")
        out_state[k] = out_state[k].to(states[0][k].dtype).contiguous()

    r_new = sum(c["r"] for c in cfgs)
    cfg = dict(ref)
    cfg["r"] = r_new
    cfg["lora_alpha"] = ref["lora_alpha"] * r_new // ref["r"]   # keep alpha/r fixed
    cfg["lora_dropout"] = 0.0                                    # inference-only artifact

    os.makedirs(args.out, exist_ok=True)
    save_file(out_state, os.path.join(args.out, "adapter_model.safetensors"))
    json.dump(cfg, open(os.path.join(args.out, "adapter_config.json"), "w"), indent=2)
    for extra in ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
                  "preprocessor_config.json", "chat_template.jinja"):
        src = os.path.join(parts[0][0], extra)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, extra))

    mix = "  ".join(f"{os.path.basename(p)}*{c:g}" for p, c in parts)
    print(f"[task_vector] {mix}  ->  {args.out}")
    print(f"[task_vector] r {ref['r']} -> {r_new}, alpha {ref['lora_alpha']} -> "
          f"{cfg['lora_alpha']} (scaling {cfg['lora_alpha']/r_new:g} unchanged)")


if __name__ == "__main__":
    main()
