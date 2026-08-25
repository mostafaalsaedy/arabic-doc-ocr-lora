@echo off
rem Stage-6 pipeline: HTML table retargeting for TEDS. Continues eyoun-soup50 (champion).
rem Context: first TEDS measurement put eyoun-soup50 at 0.0 (emits markdown tables, the
rem benchmark scores HTML structure). The free output converter lifts that to 34.3 vs
rem Baseer 66, but 18/58 table pages still score 0 because CELLS ARE MIS-SEGMENTED —
rem that is what this stage trains. ~18k table rows already exist in data/unified and
rem are converted markdown->HTML on the fly (no shard rebuild).
rem RELAUNCH after a crash: auto-resumes from the latest stage6 checkpoint.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
echo [%date% %time%] stage6 pipeline started >> pipeline.log

.venv\Scripts\python.exe train\stage6.py >> train\stage6_run.log 2>&1
echo [%date% %time%] stage6 training exited >> pipeline.log

if not exist train\adapters\eyoun-s6\adapter_model.safetensors (
  echo [%date% %time%] stage6 pipeline ABORTED - no eyoun-s6 adapter, training failed >> pipeline.log
  exit /b 1
)

rem eval eyoun-s6. misraj FIRST (TEDS is the point of this stage). NOTE: no
rem --md_tables_to_html here — after training the model should emit HTML natively,
rem and forcing the converter would mask whether that actually happened.
set MB=train\merged\eyoun-s2-merged-16bit
set AD=train\adapters\eyoun-s6
set DEC=--repetition_penalty 1.2 --no_repeat_ngram_size 6 --max_pixels 1003520
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-s6 --suite misraj_dococr %DEC% >> eval\eyoun_s6_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-s6 --suite arocrbench_arabicocr --limit 50 %DEC% >> eval\eyoun_s6_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-s6 --suite arocrbench_hindawi,arocrbench_historyar,arocrbench_khattparagraph --limit 200 %DEC% >> eval\eyoun_s6_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-s6 --suite sedra_handwritten %DEC% >> eval\eyoun_s6_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-s6 --suite arocrbench_patsocr,arocrbench_isippt,arocrbench_synthesizear --limit 500 %DEC% >> eval\eyoun_s6_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter %AD% --name eyoun-s6 --suite nakba_test --limit 300 %DEC% >> eval\eyoun_s6_run.log 2>&1
echo [%date% %time%] stage6 pipeline COMPLETE (eyoun-s6 evaled) >> pipeline.log
