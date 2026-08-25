@echo off
rem Detached Stage-2 pipeline: merge shards -> Stage 2 training -> eyoun-s2 eval.
rem PRECONDITIONS: eyoun-s1 eval finished (GPU free) + DOMAIN_WEIGHT in
rem train\stage2.py reviewed against the eval gaps.
rem RELAUNCH after a crash: comment out the merge line (deterministic but slow);
rem training auto-resumes from the last stage2 checkpoint.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
rem Tier-1 OOM fix (2026-07-08): step-9357 crashes were CUDA allocator
rem FRAGMENTATION (died at ~5GB used on a 12GB card, 163->2 it/s thrash), NOT a
rem large sample - all imgs are pre-capped to 639 vis-tokens. expandable_segments
rem is unsupported on Windows; these two knobs are the Windows-compatible fix.
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
echo [%date% %time%] stage2 pipeline started >> pipeline.log

rem merge already done 2026-07-07 (commented out for crash relaunch 2026-07-08, OOM at step 9357)
rem .venv\Scripts\python.exe scripts\build_unified_dataset.py --merge >> data\merge_stage2.log 2>&1
rem echo [%date% %time%] stage2 merge done >> pipeline.log

.venv\Scripts\python.exe train\stage2.py >> train\stage2_run.log 2>&1
echo [%date% %time%] stage2 training exited >> pipeline.log

rem Guard: only eval if training actually produced the adapter (added 2026-07-08
rem after two crashes chained into eval and logged bogus COMPLETE lines).
if not exist train\adapters\eyoun-s2\adapter_model.safetensors (
  echo [%date% %time%] stage2 pipeline ABORTED - no eyoun-s2 adapter, training failed >> pipeline.log
  exit /b 1
)

rem --- eval eyoun-s2 with the SAME caps as baseline/eyoun-s1 ---
.venv\Scripts\python.exe eval\run_eval.py --model base_model --adapter train\adapters\eyoun-s2 --name eyoun-s2 --suite arocrbench_arabicocr --limit 50 >> eval\eyoun_s2_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model base_model --adapter train\adapters\eyoun-s2 --name eyoun-s2 --suite arocrbench_hindawi,arocrbench_historyar,arocrbench_khattparagraph --limit 200 >> eval\eyoun_s2_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model base_model --adapter train\adapters\eyoun-s2 --name eyoun-s2 --suite arocrbench_patsocr,arocrbench_isippt,arocrbench_synthesizear --limit 500 >> eval\eyoun_s2_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model base_model --adapter train\adapters\eyoun-s2 --name eyoun-s2 --suite nakba_test --limit 300 >> eval\eyoun_s2_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model base_model --adapter train\adapters\eyoun-s2 --name eyoun-s2 --suite sedra_handwritten,misraj_dococr >> eval\eyoun_s2_run.log 2>&1
echo [%date% %time%] stage2 pipeline COMPLETE (eyoun-s2 evaled) >> pipeline.log
