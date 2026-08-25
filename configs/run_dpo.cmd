@echo off
rem DPO anti-hallucination pipeline: preference pairs (faithful vs drifted OCR)
rem on the eyoun-s4 champion. RELAUNCH after a crash: auto-resumes from dpo ckpt.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
echo [%date% %time%] dpo pipeline started >> pipeline.log

.venv\Scripts\python.exe train\dpo.py >> train\dpo_run.log 2>&1
echo [%date% %time%] dpo training exited >> pipeline.log

if not exist train\adapters\eyoun-dpo\adapter_model.safetensors (
  echo [%date% %time%] dpo pipeline ABORTED - no eyoun-dpo adapter, training failed >> pipeline.log
  exit /b 1
)

rem eval eyoun-dpo: SAME caps + anti-loop decode + hi-res. Order: the behavioral
rem suites first — misraj (loop pages), khatt (hallucination), nakba (in-domain),
rem sedra (protection check).
set MB=train\merged\eyoun-s2-merged-16bit
set DEC=--repetition_penalty 1.2 --no_repeat_ngram_size 6 --max_pixels 1003520
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-dpo --name eyoun-dpo --suite misraj_dococr %DEC% >> eval\eyoun_dpo_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-dpo --name eyoun-dpo --suite arocrbench_khattparagraph --limit 200 %DEC% >> eval\eyoun_dpo_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-dpo --name eyoun-dpo --suite nakba_test --limit 300 %DEC% >> eval\eyoun_dpo_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-dpo --name eyoun-dpo --suite sedra_handwritten %DEC% >> eval\eyoun_dpo_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-dpo --name eyoun-dpo --suite arocrbench_arabicocr --limit 50 %DEC% >> eval\eyoun_dpo_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-dpo --name eyoun-dpo --suite arocrbench_hindawi,arocrbench_historyar --limit 200 %DEC% >> eval\eyoun_dpo_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-dpo --name eyoun-dpo --suite arocrbench_patsocr,arocrbench_isippt,arocrbench_synthesizear --limit 500 %DEC% >> eval\eyoun_dpo_run.log 2>&1
echo [%date% %time%] dpo pipeline COMPLETE (eyoun-dpo evaled) >> pipeline.log
