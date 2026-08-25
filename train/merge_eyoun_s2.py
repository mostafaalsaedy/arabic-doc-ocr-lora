"""Merge the eyoun-s2 LoRA into the base -> a standalone 16-bit model.

Stage 3 needs this because it UNFREEZES vision (fresh vision+language LoRA), which
can't cleanly continue eyoun-s2's language-only adapter — so we bake eyoun-s2 into the
weights and start a new adapter on top. The merged model is also the Phase-6
packaging artifact (GGUF source).

    python train/merge_eyoun_s2.py            # -> train/merged/eyoun-s2-merged-16bit

Idempotent: skips if the output already exists (delete to force a rebuild).
"""
import os

BASE = os.environ.get("EYOUN_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S2_ADAPTER = os.path.join(BASE, "train", "adapters", "eyoun-s2")
OUT = os.path.join(BASE, "train", "merged", "eyoun-s2-merged-16bit")


def main():
    if os.path.exists(os.path.join(OUT, "config.json")):
        print(f"merged model already exists -> {OUT} (delete to rebuild)")
        return
    from unsloth import FastVisionModel

    # load in 16-bit (not 4-bit) so the merge doesn't bake in quantization error
    model, processor = FastVisionModel.from_pretrained(
        S2_ADAPTER, load_in_4bit=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model.save_pretrained_merged(OUT, processor, save_method="merged_16bit")
    print(f"merged 16-bit model saved -> {OUT}")


if __name__ == "__main__":
    main()
