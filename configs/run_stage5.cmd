@echo off
rem Stage-5 pipeline: FIRST real handwritten-paragraph data (khatt lever) + fresh
rem img2md book pages (misraj lever). Continues eyoun-s4 (champion; eyoun-s4-1 rejected).
rem RELAUNCH after a crash: auto-resumes from the latest stage5 checkpoint.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
echo [%date% %time%] stage5 pipeline started >> pipeline.log

.venv\Scripts\python.exe train\stage5.py >> train\stage5_run.log 2>&1
echo [%date% %time%] stage5 training exited >> pipeline.log

if not exist train\adapters\eyoun-s5\adapter_model.safetensors (
  echo [%date% %time%] stage5 pipeline ABORTED - no eyoun-s5 adapter, training failed >> pipeline.log
  exit /b 1
)

rem eval eyoun-s5: SAME caps + anti-loop decode + hi-res; khatt + misraj + sedra FIRST
rem (khatt is the Stage-5 lever). Base = MERGED eyoun-s2 (eyoun-s4 chain).
set MB=train\merged\eyoun-s2-merged-16bit
set DEC=--repetition_penalty 1.2 --no_repeat_ngram_size 6 --max_pixels 1003520
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s5 --name eyoun-s5 --suite arocrbench_khattparagraph --limit 200 %DEC% >> eval\eyoun_s5_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s5 --name eyoun-s5 --suite misraj_dococr %DEC% >> eval\eyoun_s5_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s5 --name eyoun-s5 --suite sedra_handwritten %DEC% >> eval\eyoun_s5_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s5 --name eyoun-s5 --suite arocrbench_arabicocr --limit 50 %DEC% >> eval\eyoun_s5_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s5 --name eyoun-s5 --suite arocrbench_hindawi,arocrbench_historyar --limit 200 %DEC% >> eval\eyoun_s5_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s5 --name eyoun-s5 --suite arocrbench_patsocr,arocrbench_isippt,arocrbench_synthesizear --limit 500 %DEC% >> eval\eyoun_s5_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s5 --name eyoun-s5 --suite nakba_test --limit 300 %DEC% >> eval\eyoun_s5_run.log 2>&1
echo [%date% %time%] stage5 pipeline COMPLETE (eyoun-s5 evaled) >> pipeline.log
