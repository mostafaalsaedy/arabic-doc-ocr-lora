@echo off
rem Sweep 3 (2026-08-06): re-run with the STRATIFIED misraj slice (norm + loop tail).
rem Purpose 1 = VALIDATE the fixed harness: eyoun-s5 (λ=1.0) is KNOWN to be 1.10x worse
rem   on full-suite misraj, and the old first-40 slice reported a harmless 0.99x.
rem   The loop-tail slice must now SHOW that damage, or the harness is still blind.
rem Purpose 2 = if validated, re-judge λ=0.75 (which looked best on the blind slice).
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
set PY=.venv\Scripts\python.exe
echo [%date% %time%] sweep3 started >> pipeline.log

rem new baseline on the stratified slice set (eyoun-s4 = the reference)
%PY% scripts\dev_eval.py --adapter train\adapters\eyoun-s4 --name v3_lam0.00 >> eval\sweep3.log 2>&1
echo [%date% %time%] sweep3: v3_lam0.00 done >> pipeline.log

rem CALIBRATION: known-bad point. loop-tail must degrade vs baseline.
%PY% scripts\dev_eval.py --adapter train\adapters\eyoun-s5 --name v3_s5_1.00 --baseline v3_lam0.00 >> eval\sweep3.log 2>&1
echo [%date% %time%] sweep3: v3_s5_1.00 (calibration) done >> pipeline.log

rem current champion, for a like-for-like reference on the new slice set
%PY% scripts\dev_eval.py --adapter train\adapters\eyoun-soup50 --name v3_soup50 --baseline v3_lam0.00 >> eval\sweep3.log 2>&1
echo [%date% %time%] sweep3: v3_soup50 done >> pipeline.log

rem the candidate that looked best on the blind slice — re-judge it honestly
%PY% scripts\dev_eval.py --adapter train\adapters\_mix_s5075 --name v3_s5_0.75 --baseline v3_lam0.00 >> eval\sweep3.log 2>&1
echo [%date% %time%] sweep3: v3_s5_0.75 done >> pipeline.log

echo [%date% %time%] sweep3 COMPLETE >> pipeline.log
