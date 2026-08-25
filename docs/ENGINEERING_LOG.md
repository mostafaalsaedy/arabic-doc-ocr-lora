# Engineering log

A record of what was tried over one training cycle, including the five stages that were
rejected. Kept public because the failures cost more time than the successes and are the
more useful half of the project.

Hardware throughout: one RTX 5070 Ti Laptop GPU, 12 GB VRAM. Base model:
Qwen2.5-VL-3B-Instruct, QLoRA, r=64 / alpha=128.

## Adapter lineage

| adapter | what it was | verdict |
|---|---|---|
| stage 1 … 3.1 | progressive SFT on the unified corpus | superseded |
| **stage 4** | real-print document mix | ex-champion; still the best single *trained* adapter |
| stage 4.1 | second epoch of the same mix | ❌ wash — and its task vector is actively harmful in combinations |
| stage 5 | + real KHATT paragraphs, + 13.7k img2md book pages | ❌ rejected at full strength — but half of it is in the champion |
| DPO v1 / v2 | anti-hallucination preference training | ❌❌ rejected; v2 was total model collapse |
| soup-50 | 0.5 · stage4 + 0.5 · stage5 | ex-champion |
| stage 6 / 6.1 | HTML table structure training | ❌ rejected — but 6.1 is in the champion |
| **soup-61** | 0.5 · soup-50 + 0.5 · stage6.1 | ★ **champion**, built with zero training |

Final: 6 wins, 4 flat, zero regressions against the previous champion across the full
10-suite evaluation.

## Rules learned the expensive way

**1. DPO is closed.** Two runs, two different catastrophic failure modes. A λ-sweep proved
the learned direction carried no recoverable signal — the target suite never moved at any
dose. Not worth reopening without genuinely new evidence.

**2. More SFT data is a dead lever.** Stages 4.1 and 5 both demonstrated it independently.
At this scale, gains come from *representation* (how the target is formatted) or from
*weight arithmetic* — not from volume.

**3. A rejected stage is not worthless.** λ-sweep its task vector before writing it off.
That is exactly how a fully rejected stage 5 became half of the champion. The exception:
second-epoch stages, trained on already-seen data, were harmful in every combination tried.

**4. Evaluate raw and post-processed separately, per candidate.** The champion scores TEDS
2.91 raw and 18.34 with markdown→HTML table conversion. Another candidate is the opposite —
15.7 native, 13.5 converted. Judging on a single mode would have rejected the right model.

**5. Small dev slices cannot rank the hard suite.** Proven twice. The failure set is
adapter-specific — two adapters shared only 2 of their ~15 failing pages — so any fixed
index set misses the next candidate's novel failures. Decisions on that suite need the full
400-page run (~12 h). Small slices remain useful as a *collapse detector*.

**6. Training metrics do not predict evaluation collapse** at small data scale. The DPO v2
run had healthy margins and accuracy while destroying the model. Evaluate mid-run; do not
trust the loss curve.

**7. Checkpoint resume needs `ignore_data_skip=True`** plus per-step `empty_cache` on a
12 GB card, or fragmentation kills the run.

**8. `save_pretrained_merged` can silently no-op.** The unsloth helper copied the base
weights without applying the LoRA, and peft's `merge_and_unload` was OOM-killed at this
size. `train/merge_champion.py` is a streaming merger with a built-in "did the weights
actually change" assertion — the only approach that worked here.

## Tooling built this cycle

| script | what it does |
|---|---|
| `tools/task_vector.py` | exact LoRA combination `W = Σ cᵢ·(Bᵢ@Aᵢ)`, `Σcᵢ = 1`, by rank-axis concatenation. Naive elementwise blending is wrong — it introduces cross terms. Verified to ~1e-10. |
| `eval/dev_eval.py` | ~160-sample dev slice, cheap→expensive ordering, with a collapse canary that aborts a bad candidate in ~6 minutes. |
| `tools/teds.py` | TEDS (PubTabNet definition, Zhang-Shasha tree edit distance), pure stdlib, no dependencies. |
| `tools/postprocess.py` | `md_tables_to_html` and its inverse, making output table format an inference-time toggle rather than a training target. |

## The gap to the reference model

Word Error Rate 0.533 against the reference Arabic doc-OCR model's 0.25 — improved from
0.560 across the whole cycle, and all of that movement came from souping and
post-processing rather than from training.

Table training failed twice for a structural reason: the available supervision is roughly
four times noisier than the benchmark (16% ragged tables against 4%), so it degrades cell
segmentation more than it helps. Even perfect table handling projects to only ~0.48 WER,
and non-table pages sit well above the reference on their own.

The reference is a 7B model trained with far more data and compute. This is a 3B QLoRA
trained on a single laptop GPU. The remaining gap is a capability gap, not a formatting
one — treating it as reachable by tuning would have been the next expensive mistake.

## Release

Merged to a standalone model, converted to GGUF (f16, Q8_0, Q4_K_M) with the vision
projector, and validated end to end with `llama-mtmd-cli` on a real benchmark page before
shipping. Both the language model and the `mmproj` projector are required — without the
projector the model silently degrades to text-only.
