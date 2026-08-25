@echo off
rem Stage-4 pipeline: real-print doc refinement (presightai 32k pages) to close
rem the misraj gap. Continues the eyoun-s3b adapter; protects sedra (hw_line 1.0
rem reminder). NO merge / NO shard rebuild needed (presightai shard built +
rem merged 2026-07-19). RELAUNCH after a crash: training auto-resumes from the
rem last stage4 checkpoint.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
echo [%date% %time%] stage4 pipeline started >> pipeline.log

.venv\Scripts\python.exe train\stage4.py >> train\stage4_run.log 2>&1
echo [%date% %time%] stage4 training exited >> pipeline.log

if not exist train\adapters\eyoun-s4\adapter_model.safetensors (
  echo [%date% %time%] stage4 pipeline ABORTED - no eyoun-s4 adapter, training failed >> pipeline.log
  exit /b 1
)

rem eval eyoun-s4: SAME caps + anti-loop decode + hi-res. eyoun-s4 continues eyoun-s3b whose
rem base is the MERGED eyoun-s2 — so eval loads that base + the eyoun-s4 adapter.
set MB=train\merged\eyoun-s2-merged-16bit
set DEC=--repetition_penalty 1.2 --no_repeat_ngram_size 6 --max_pixels 1003520
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s4 --name eyoun-s4 --suite misraj_dococr %DEC% >> eval\eyoun_s4_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s4 --name eyoun-s4 --suite sedra_handwritten %DEC% >> eval\eyoun_s4_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s4 --name eyoun-s4 --suite arocrbench_arabicocr --limit 50 %DEC% >> eval\eyoun_s4_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s4 --name eyoun-s4 --suite arocrbench_hindawi,arocrbench_historyar,arocrbench_khattparagraph --limit 200 %DEC% >> eval\eyoun_s4_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s4 --name eyoun-s4 --suite arocrbench_patsocr,arocrbench_isippt,arocrbench_synthesizear --limit 500 %DEC% >> eval\eyoun_s4_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s4 --name eyoun-s4 --suite nakba_test --limit 300 %DEC% >> eval\eyoun_s4_run.log 2>&1
echo [%date% %time%] stage4 pipeline COMPLETE (eyoun-s4 evaled) >> pipeline.log
