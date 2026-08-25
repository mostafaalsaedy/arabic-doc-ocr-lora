@echo off
rem DPO v2 pipeline: ON-POLICY pairs (eyoun-s4's own loop/hallucination outputs on
rem training images) + beta 0.05 + rpo_alpha 1.0 anchor + lr 3e-6. v1 (off-policy
rem nakba pairs) collapsed decode - see memory qwen-vl-dpo-plumbing.
rem RELAUNCH after a crash: auto-resumes from dpo2 ckpt.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
echo [%date% %time%] dpo2 pipeline started >> pipeline.log

.venv\Scripts\python.exe train\dpo.py >> train\dpo2_run.log 2>&1
echo [%date% %time%] dpo2 training exited >> pipeline.log

if not exist train\adapters\eyoun-dpo2\adapter_model.safetensors (
  echo [%date% %time%] dpo2 pipeline ABORTED - no eyoun-dpo2 adapter, training failed >> pipeline.log
  exit /b 1
)

rem eval eyoun-dpo2: max_new_tokens CAPPED at 1024 so loop-pages are bounded (~35s
rem not 50min). Cap also applied to nothing else - suites/caps match prior evals.
set MB=train\merged\eyoun-s2-merged-16bit
set DEC=--repetition_penalty 1.2 --no_repeat_ngram_size 6 --max_pixels 1003520 --max_new_tokens 1024
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-dpo2 --name eyoun-dpo2 --suite misraj_dococr %DEC% >> eval\eyoun_dpo2_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-dpo2 --name eyoun-dpo2 --suite arocrbench_khattparagraph --limit 200 %DEC% >> eval\eyoun_dpo2_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-dpo2 --name eyoun-dpo2 --suite sedra_handwritten %DEC% >> eval\eyoun_dpo2_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-dpo2 --name eyoun-dpo2 --suite nakba_test --limit 300 %DEC% >> eval\eyoun_dpo2_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-dpo2 --name eyoun-dpo2 --suite arocrbench_arabicocr --limit 50 %DEC% >> eval\eyoun_dpo2_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-dpo2 --name eyoun-dpo2 --suite arocrbench_hindawi,arocrbench_historyar --limit 200 %DEC% >> eval\eyoun_dpo2_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-dpo2 --name eyoun-dpo2 --suite arocrbench_patsocr,arocrbench_isippt,arocrbench_synthesizear --limit 500 %DEC% >> eval\eyoun_dpo2_run.log 2>&1
echo [%date% %time%] dpo2 pipeline COMPLETE (eyoun-dpo2 evaled) >> pipeline.log
