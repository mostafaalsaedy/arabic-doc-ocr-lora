@echo off
rem Detached pipeline: remaining baselines -> Stage 1 training.
rem Safe to re-run: evals merge into summary.json, training auto-resumes from last checkpoint.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
echo [%date% %time%] pipeline started >> pipeline.log

.venv\Scripts\python.exe eval\run_eval.py --model base_model --name stock-qwen25vl-3b --suite sedra_handwritten,misraj_dococr >> eval\baseline_run4.log 2>&1
echo [%date% %time%] evals A done >> pipeline.log

.venv\Scripts\python.exe eval\run_eval.py --model base_model --name stock-qwen25vl-3b --suite nakba_test --limit 300 >> eval\baseline_run5.log 2>&1
echo [%date% %time%] evals B done >> pipeline.log

.venv\Scripts\python.exe train\stage1.py >> train\stage1_run.log 2>&1
echo [%date% %time%] stage1 exited >> pipeline.log
