@echo off
rem Sweep 2 (2026-08-06): push the soup ratio past the confirmed λ=0.50 win, and
rem test the two task vectors never tried (eyoun-s4-1). Dev slice set v2 = the three
rem RELIABLE suites only (arabicocr/hindawi/misraj); sedra+khatt dropped after
rem sweep-1 showed their dev numbers do not predict full-suite results.
rem Reference points already measured on this slice set:
rem   eyoun-s4    (lam0.00): arabicocr 0.0402  hindawi 0.1369  misraj 0.3135
rem   soup50   (s5 0.50): arabicocr 0.0347  hindawi 0.1218  misraj 0.3144
rem Binding constraint: FULL eyoun-s5 (λ=1) hurt misraj 1.10x, so watch misraj.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
set PY=.venv\Scripts\python.exe
echo [%date% %time%] sweep2 started >> pipeline.log

%PY% scripts\task_vector.py --mix eyoun-s4:0.35 --mix eyoun-s5:0.65 --out train\adapters\_mix_s5065 >> eval\sweep2.log 2>&1
%PY% scripts\dev_eval.py --adapter train\adapters\_mix_s5065 --name s5_0.65 --baseline lam0.00 >> eval\sweep2.log 2>&1
echo [%date% %time%] sweep2: s5_0.65 done >> pipeline.log

%PY% scripts\task_vector.py --mix eyoun-s4:0.25 --mix eyoun-s5:0.75 --out train\adapters\_mix_s5075 >> eval\sweep2.log 2>&1
%PY% scripts\dev_eval.py --adapter train\adapters\_mix_s5075 --name s5_0.75 --baseline lam0.00 >> eval\sweep2.log 2>&1
echo [%date% %time%] sweep2: s5_0.75 done >> pipeline.log

rem 3-way: keep the winning s5 dose, spend a little mass on the untested eyoun-s4-1
rem task vector (Stage 4.1 was a wash at full strength — same situation eyoun-s5 was
rem in before dilution, so it is worth one cheap probe).
%PY% scripts\task_vector.py --mix eyoun-s4:0.40 --mix eyoun-s5:0.50 --mix eyoun-s4-1:0.10 --out train\adapters\_mix_soup3 >> eval\sweep2.log 2>&1
%PY% scripts\dev_eval.py --adapter train\adapters\_mix_soup3 --name soup3 --baseline lam0.00 >> eval\sweep2.log 2>&1
echo [%date% %time%] sweep2: soup3 done >> pipeline.log

echo [%date% %time%] sweep2 COMPLETE >> pipeline.log
