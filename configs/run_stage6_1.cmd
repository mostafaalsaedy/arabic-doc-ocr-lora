@echo off
rem Stage-6.1: HTML table retargeting at a CORRECTED dose.
rem Stage 6 used TABLE_WEIGHT 2.0 -> tables were ~64%% of the mix vs ~14.5%% of real
rem doc pages. It FIXED recognition (TEDS-0 pages 18/58 -> 6/58) but hallucinated
rem tables on 40 clean pages, so net TEDS fell 18.7 -> 15.6 and misraj CER worsened
rem 0.387 -> 0.409. 6.1 sets TABLE_WEIGHT 0.45 (~22%% of the mix) to keep the
rem recognition gain without the over-trigger. Continues eyoun-soup50 (champion).
rem RELAUNCH after a crash: auto-resumes from the latest stage6_1 checkpoint.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
echo [%date% %time%] stage6_1 pipeline started >> pipeline.log

.venv\Scripts\python.exe train\stage6_1.py >> train\stage6_1_run.log 2>&1
echo [%date% %time%] stage6_1 training exited >> pipeline.log

if not exist train\adapters\eyoun-s6-1\adapter_model.safetensors (
  echo [%date% %time%] stage6_1 pipeline ABORTED - no eyoun-s6-1 adapter >> pipeline.log
  exit /b 1
)

rem misraj FIRST (TEDS is the point). No --md_tables_to_html: the model should emit
rem HTML natively now, and forcing the converter would mask whether it actually did.
set MB=train\merged\eyoun-s2-merged-16bit
set AD=train\adapters\eyoun-s6-1
set DEC=--repetition_penalty 1.2 --no_repeat_ngram_size 6 --max_pixels 1003520
set EL=eval\eyoun_s6_1_run.log
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-s6-1 --suite misraj_dococr %DEC% >> %EL% 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-s6-1 --suite arocrbench_arabicocr --limit 50 %DEC% >> %EL% 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-s6-1 --suite arocrbench_hindawi,arocrbench_historyar,arocrbench_khattparagraph --limit 200 %DEC% >> %EL% 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-s6-1 --suite sedra_handwritten %DEC% >> %EL% 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-s6-1 --suite arocrbench_patsocr,arocrbench_isippt,arocrbench_synthesizear --limit 500 %DEC% >> %EL% 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-s6-1 --suite nakba_test --limit 300 %DEC% >> %EL% 2>&1
echo [%date% %time%] stage6_1 pipeline COMPLETE (eyoun-s6-1 evaled) >> pipeline.log
