# Evaluation results

One `summary.json` per adapter, as produced by `eval/run_eval.py`. Every number in the
README and the model card comes from these files.

`stock-qwen25vl-3b` is the un-finetuned baseline. `eyoun-soup61` is the shipped champion.
`eyoun-s*` are the sequential SFT stages; `eyoun-soup*` are task-vector combinations built with
`tools/task_vector.py`; `eyoun-dpo2` is the failed preference-training run, kept for the
record.

Metrics: `cer` and `wer` are character and word error rate (lower is better), `n` is the
number of pages scored, and `teds` is table structure similarity on the subset of pages
containing tables (`teds_n`), higher is better.

Note that `eyoun-soup61`'s `misraj_dococr` TEDS reads 2.91 here because these runs are scored
**raw**. The same output through `tools/postprocess.py::md_tables_to_html` scores 18.34.
Both numbers are real; the converter is part of the shipped inference configuration. See
rule 4 in the engineering log.

Per-page prediction files are not included — they embed benchmark ground truth from
third-party datasets, which are distributed under their own licences.
