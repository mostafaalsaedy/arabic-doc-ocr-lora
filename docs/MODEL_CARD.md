# Eyoun-3B — Arabic document OCR (GGUF)

Qwen2.5-VL-3B-Instruct fine-tuned for **faithful Arabic OCR** — transcribe what is on the
page, do not translate or paraphrase. Trained locally as a QLoRA on a single 12 GB
RTX 5070 Ti Laptop GPU.

## Files

| file | size | what |
|---|---|---|
| `Eyoun-3B-Q4_K_M.gguf` | 1.8 GB | smallest — fine for most use |
| `Eyoun-3B-Q8_0.gguf` | 3.1 GB | **recommended** — near-lossless |
| `Eyoun-3B-f16.gguf` | 5.8 GB | full precision, for re-quantising |
| `mmproj-Eyoun-3B-f16.gguf` | 1.3 GB | **required** — vision projector; without it the model is text-only |
| `postprocess.py` | — | table format converter, see below |
| `teds.py` | — | TEDS metric (table structure scoring) |

Pick Q8_0 if you have the memory; Q4_K_M if you don't. We spot-checked both on the same page
and did **not** observe a systematic quality difference — each made a different single-digit
error. The suite numbers below were measured on the unquantised model.

## Use

LM Studio: drop the folder in your models directory — it picks up `mmproj-*.gguf`
automatically. Or with llama.cpp:

```
llama-mtmd-cli -m Eyoun-3B-Q8_0.gguf \
               --mmproj mmproj-Eyoun-3B-f16.gguf \
               --image page.png -p "استخرج النص من الصورة." --temp 0
```

Recommended decode: `--temp 0`, `repetition_penalty 1.2`, `no_repeat_ngram_size 6`.
The repetition controls matter — without them the model can loop on dense pages.

## ⚠️ Tables: run the post-processor

The model emits **markdown** pipe tables. If your target format is HTML `<table>` (as the
misraj/Baseer benchmark uses), pass the output through `postprocess.md_tables_to_html`.
This is not cosmetic — on the benchmark it is the difference between **TEDS 2.9 and 18.3**.
The reverse (`html_tables_to_md`) is also provided, so output format is your choice.

```python
from postprocess import md_tables_to_html
text = md_tables_to_html(model_output)
```

## Evaluation

Character Error Rate, lower is better. `stock` = un-finetuned Qwen2.5-VL-3B.

| suite | stock | Eyoun-3B |
|---|---|---|
| arocrbench_arabicocr | 1.120 | **0.035** |
| arocrbench_patsocr | 1.329 | **0.032** |
| arocrbench_synthesizear | 1.445 | **0.066** |
| arocrbench_isippt | 2.480 | **0.077** |
| arocrbench_hindawi | 0.583 | **0.126** |
| nakba_test (newspaper) | 1.862 | **0.191** |
| sedra_handwritten | 4.241 | **0.347** |
| arocrbench_historyar | 3.061 | **0.360** |
| misraj_dococr | 1.148 | **0.359** |
| arocrbench_khattparagraph | 3.297 | 4.141 ⚠️ |

misraj_dococr (the competitive benchmark): **CER 0.359 · WER 0.533 · TEDS 18.3**
(with the post-processor).

## Honest limitations

* **Handwritten paragraphs (khatt) are worse than the base model.** CER 4.14 vs stock's
  3.30 — the model hallucinates on dense multi-line handwriting. Five training stages
  targeting this failed, including one on 2,160 real KHATT paragraphs. Use the stock model
  or another tool for that input type. Single-line handwriting (sedra 0.347) is fine.
* **Not competitive with Baseer**, the reference Arabic doc-OCR model (WER 0.25 / TEDS 66
  vs our 0.533 / 18.3). Baseer is a 7B model trained with far more data and compute. This
  model is a 3B QLoRA trained on one laptop GPU; treat it as a strong local/offline option,
  not a SOTA claim.
* **Table structure is mediocre.** Cell segmentation on complex tables is unreliable
  (median per-page TEDS ~27 even where it works). Text inside tables is usually recoverable;
  the grid often is not.
* **Markdown emphasis is reproduced inconsistently.** On misraj the model emits `**bold**`
  on 169/400 pages where the reference has it on 257, and over-produces `#` headings
  (109 vs 60). This is model behaviour, not a quantization artifact — it is identical at
  Q8_0 and f16. If you need faithful emphasis markup, post-edit or don't rely on it.
* **Digits are the weakest character class.** Long numeric sequences (page references,
  citation numbers) are error-prone at every precision, and the model sometimes converts
  Arabic-Indic numerals to Western ones. Verify numbers if they matter.

## Provenance

Built by task-vector arithmetic over two QLoRA fine-tunes rather than by a single training
run: `0.5·(0.5·eyoun-s4 + 0.5·eyoun-s5) + 0.5·eyoun-s6-1`, merged into the base. Training data was
~289k audited Arabic OCR rows (real print, manuscripts, handwriting, newspapers, book
pages). arocrbench/misraj/sedra/nakba evaluation sets were excluded from training, and
KHATT training pages sharing text with the benchmark were filtered out.

License follows the base model (Qwen2.5-VL-3B-Instruct) and the training corpora.
