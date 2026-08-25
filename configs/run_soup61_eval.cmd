@echo off
rem Full-suite eval of eyoun-soup61 = 0.5*eyoun-soup50 + 0.5*eyoun-s6-1 (task-vector soup,
rem NO training). Rationale: eyoun-s6-1 is a better TEXT model than the champion
rem (sedra 0.88x, hindawi 0.92x, isippt 0.92x) but a worse TABLE model
rem (misraj TEDS 15.7 vs 18.7). Souping is the one move that has already produced a
rem champion here, so test whether the blend keeps the text gains while diluting the
rem table loss. Decode flags match the champion's own eval exactly (no
rem --max_new_tokens cap) so numbers stay directly comparable.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
echo [%date% %time%] soup61 eval started >> pipeline.log

set MB=train\merged\eyoun-s2-merged-16bit
set AD=train\adapters\eyoun-soup61
set DEC=--repetition_penalty 1.2 --no_repeat_ngram_size 6 --max_pixels 1003520
set EL=eval\eyoun_soup61_run.log

rem misraj first: it carries BOTH decisive numbers (TEDS and the primary CER/WER).
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-soup61 --suite misraj_dococr %DEC% >> %EL% 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-soup61 --suite sedra_handwritten %DEC% >> %EL% 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-soup61 --suite arocrbench_hindawi,arocrbench_historyar,arocrbench_khattparagraph --limit 200 %DEC% >> %EL% 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-soup61 --suite arocrbench_arabicocr --limit 50 %DEC% >> %EL% 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-soup61 --suite arocrbench_patsocr,arocrbench_isippt,arocrbench_synthesizear --limit 500 %DEC% >> %EL% 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-soup61 --suite nakba_test --limit 300 %DEC% >> %EL% 2>&1
echo [%date% %time%] soup61 eval COMPLETE >> pipeline.log
