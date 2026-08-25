@echo off
rem Stage-3.1 pipeline: corrects the Stage-3 handwritten mistake (restore
rem handwritten_line 3x, drop synthetic handwritten_paragraph) to recover sedra
rem while keeping the hi-res misraj/historyar gains. Clean restart from the
rem eyoun-s2-merged base. NO merge / NO shard rebuild (both done; data/unified already
rem holds the hi-res shards, and dropping handwritten_paragraph is done via
rem DOMAIN_WEIGHT). RELAUNCH after a crash: training auto-resumes from stage3b ckpt.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
echo [%date% %time%] stage3.1 pipeline started >> pipeline.log

.venv\Scripts\python.exe train\stage3_1.py >> train\stage3b_run.log 2>&1
echo [%date% %time%] stage3.1 training exited >> pipeline.log

if not exist train\adapters\eyoun-s3b\adapter_model.safetensors (
  echo [%date% %time%] stage3.1 pipeline ABORTED - no eyoun-s3b adapter, training failed >> pipeline.log
  exit /b 1
)

rem eval eyoun-s3b: SAME caps + anti-loop decode + hi-res, loading the MERGED eyoun-s2 base
rem + the eyoun-s3b adapter (adapter is relative to that base, NOT base_model).
set MB=train\merged\eyoun-s2-merged-16bit
set DEC=--repetition_penalty 1.2 --no_repeat_ngram_size 6 --max_pixels 1003520
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s3b --name eyoun-s3b --suite arocrbench_arabicocr --limit 50 %DEC% >> eval\eyoun_s3b_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s3b --name eyoun-s3b --suite arocrbench_hindawi,arocrbench_historyar,arocrbench_khattparagraph --limit 200 %DEC% >> eval\eyoun_s3b_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s3b --name eyoun-s3b --suite arocrbench_patsocr,arocrbench_isippt,arocrbench_synthesizear --limit 500 %DEC% >> eval\eyoun_s3b_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s3b --name eyoun-s3b --suite nakba_test --limit 300 %DEC% >> eval\eyoun_s3b_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s3b --name eyoun-s3b --suite sedra_handwritten,misraj_dococr %DEC% >> eval\eyoun_s3b_run.log 2>&1
echo [%date% %time%] stage3.1 pipeline COMPLETE (eyoun-s3b evaled) >> pipeline.log
