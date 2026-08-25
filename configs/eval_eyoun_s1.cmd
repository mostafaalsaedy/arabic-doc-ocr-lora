@echo off
rem Phase 4: evaluate Stage-1 adapter (eyoun-s1) on all suites, mirroring baseline sample caps.
rem Safe to re-run: results merge into eval\results\eyoun-s1\summary.json.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
echo [%date% %time%] eyoun-s1 eval started >> pipeline.log

.venv\Scripts\python.exe eval\run_eval.py --model base_model --adapter train\adapters\eyoun-s1 --name eyoun-s1 --suite arocrbench_arabicocr --limit 50 >> eval\eyoun_s1_run.log 2>&1
echo [%date% %time%] eyoun-s1: arabicocr done >> pipeline.log

.venv\Scripts\python.exe eval\run_eval.py --model base_model --adapter train\adapters\eyoun-s1 --name eyoun-s1 --suite arocrbench_hindawi,arocrbench_historyar,arocrbench_khattparagraph --limit 200 >> eval\eyoun_s1_run.log 2>&1
echo [%date% %time%] eyoun-s1: hindawi/historyar/khatt done >> pipeline.log

.venv\Scripts\python.exe eval\run_eval.py --model base_model --adapter train\adapters\eyoun-s1 --name eyoun-s1 --suite arocrbench_patsocr,arocrbench_isippt,arocrbench_synthesizear --limit 500 >> eval\eyoun_s1_run.log 2>&1
echo [%date% %time%] eyoun-s1: patsocr/isippt/synthesizear done >> pipeline.log

.venv\Scripts\python.exe eval\run_eval.py --model base_model --adapter train\adapters\eyoun-s1 --name eyoun-s1 --suite nakba_test --limit 300 >> eval\eyoun_s1_run.log 2>&1
echo [%date% %time%] eyoun-s1: nakba done >> pipeline.log

.venv\Scripts\python.exe eval\run_eval.py --model base_model --adapter train\adapters\eyoun-s1 --name eyoun-s1 --suite sedra_handwritten,misraj_dococr >> eval\eyoun_s1_run.log 2>&1
echo [%date% %time%] eyoun-s1 eval COMPLETE >> pipeline.log
