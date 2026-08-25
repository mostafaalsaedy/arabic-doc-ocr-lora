@echo off
rem Full-suite validation of eyoun-soup50 = 0.5*eyoun-s4 + 0.5*eyoun-s5 (task-vector soup,
rem NO training — built by scripts/task_vector.py from two existing adapters).
rem Dev sweep (2026-08-06) showed 4 improvements / 1 flat / 1 minor cost vs eyoun-s4.
rem DECODE FLAGS MATCH eyoun-s4's ORIGINAL EVAL EXACTLY (no --max_new_tokens cap =>
rem default 2048) so numbers are directly comparable to the recorded champion.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
echo [%date% %time%] soup50 eval started >> pipeline.log

set MB=train\merged\eyoun-s2-merged-16bit
set AD=train\adapters\eyoun-soup50
set DEC=--repetition_penalty 1.2 --no_repeat_ngram_size 6 --max_pixels 1003520

rem primary benchmark + the suites that decide champion status, first
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-soup50 --suite misraj_dococr %DEC% >> eval\eyoun_soup50_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-soup50 --suite sedra_handwritten %DEC% >> eval\eyoun_soup50_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-soup50 --suite arocrbench_arabicocr --limit 50 %DEC% >> eval\eyoun_soup50_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-soup50 --suite arocrbench_hindawi,arocrbench_historyar,arocrbench_khattparagraph --limit 200 %DEC% >> eval\eyoun_soup50_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-soup50 --suite arocrbench_patsocr,arocrbench_isippt,arocrbench_synthesizear --limit 500 %DEC% >> eval\eyoun_soup50_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-soup50 --suite nakba_test --limit 300 %DEC% >> eval\eyoun_soup50_run.log 2>&1
echo [%date% %time%] soup50 eval COMPLETE >> pipeline.log
