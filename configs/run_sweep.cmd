@echo off
rem Task-vector lambda sweep over the REJECTED challengers (2026-08-05).
rem   W(l) = W_s4 + l*(W_x - W_s4),  built EXACTLY by scripts/task_vector.py
rem Tests whether eyoun-dpo2 / eyoun-s5 hold recoverable signal beneath their damage.
rem No training — pure weight arithmetic + ~160-sample dev slices with a
rem collapse canary (arabicocr) that aborts a bad lambda in ~1 min.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
set PY=.venv\Scripts\python.exe
echo [%date% %time%] sweep started >> pipeline.log

rem ---- baseline: the champion itself (defines the dev reference numbers) ----
%PY% scripts\dev_eval.py --adapter train\adapters\eyoun-s4 --name lam0.00 >> eval\sweep.log 2>&1
echo [%date% %time%] sweep: lam0.00 (eyoun-s4 baseline) done >> pipeline.log

rem ---- eyoun-dpo2 task vector: anti-hallucination signal vs collapse damage ----
%PY% scripts\task_vector.py --mix eyoun-s4:0.90 --mix eyoun-dpo2:0.10 --out train\adapters\_mix_dpo010 >> eval\sweep.log 2>&1
%PY% scripts\dev_eval.py --adapter train\adapters\_mix_dpo010 --name dpo0.10 --baseline lam0.00 >> eval\sweep.log 2>&1
echo [%date% %time%] sweep: dpo0.10 done >> pipeline.log

%PY% scripts\task_vector.py --mix eyoun-s4:0.80 --mix eyoun-dpo2:0.20 --out train\adapters\_mix_dpo020 >> eval\sweep.log 2>&1
%PY% scripts\dev_eval.py --adapter train\adapters\_mix_dpo020 --name dpo0.20 --baseline lam0.00 >> eval\sweep.log 2>&1
echo [%date% %time%] sweep: dpo0.20 done >> pipeline.log

%PY% scripts\task_vector.py --mix eyoun-s4:0.65 --mix eyoun-dpo2:0.35 --out train\adapters\_mix_dpo035 >> eval\sweep.log 2>&1
%PY% scripts\dev_eval.py --adapter train\adapters\_mix_dpo035 --name dpo0.35 --baseline lam0.00 >> eval\sweep.log 2>&1
echo [%date% %time%] sweep: dpo0.35 done >> pipeline.log

rem ---- eyoun-s5 task vector: it won hindawi outright; is there a safe dose? ----
%PY% scripts\task_vector.py --mix eyoun-s4:0.75 --mix eyoun-s5:0.25 --out train\adapters\_mix_s5025 >> eval\sweep.log 2>&1
%PY% scripts\dev_eval.py --adapter train\adapters\_mix_s5025 --name s5_0.25 --baseline lam0.00 >> eval\sweep.log 2>&1
echo [%date% %time%] sweep: s5_0.25 done >> pipeline.log

%PY% scripts\task_vector.py --mix eyoun-s4:0.50 --mix eyoun-s5:0.50 --out train\adapters\_mix_s5050 >> eval\sweep.log 2>&1
%PY% scripts\dev_eval.py --adapter train\adapters\_mix_s5050 --name s5_0.50 --baseline lam0.00 >> eval\sweep.log 2>&1
echo [%date% %time%] sweep: s5_0.50 done >> pipeline.log

echo [%date% %time%] sweep COMPLETE >> pipeline.log
