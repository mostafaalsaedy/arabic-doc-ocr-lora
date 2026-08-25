@echo off
rem Detached Stage-3 pipeline: merge eyoun-s2 -> rebuild hi-res doc/manuscript shards
rem -> Stage 3 training (vision unfrozen) -> eyoun-s3 eval WITH anti-loop decode.
rem PRECONDITIONS: eyoun-s2 adapter exists; DOMAIN_WEIGHT/HIRES_PIXELS reviewed.
rem RELAUNCH after a crash: comment out the merge + shard-rebuild lines (slow, done)
rem and training auto-resumes from the last stage3 checkpoint.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
rem Carry the Stage-2 allocator fix (expandable_segments is unsupported on Windows).
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
echo [%date% %time%] stage3 pipeline started >> pipeline.log

rem --- 1) merge eyoun-s2 -> 16-bit base (DONE 2026-07-11, idempotent-skips; commented) ---
rem .venv\Scripts\python.exe train\merge_eyoun_s2.py >> train\stage3_run.log 2>&1
rem echo [%date% %time%] stage3 merge-eyoun-s2 done >> pipeline.log

rem --- 2) hi-res shards + okai_para + merge: DONE 2026-07-11 (data/unified is the
rem        hi-res 240k set). COMMENTED for the empty_cache/3 relaunch so --force does
rem        NOT regenerate data/unified (would reshuffle the val split). ---
rem .venv\Scripts\python.exe scripts\build_unified_dataset.py hastyle qari10k omar namaa jayanthmuthu --force >> data\build_stage3.log 2>&1
rem .venv\Scripts\python.exe scripts\build_unified_dataset.py okai_para >> data\build_stage3.log 2>&1
rem .venv\Scripts\python.exe scripts\build_unified_dataset.py --merge >> data\build_stage3.log 2>&1
rem echo [%date% %time%] stage3 shards rebuilt + merged >> pipeline.log

rem --- 3) Stage 3 training ---
.venv\Scripts\python.exe train\stage3.py >> train\stage3_run.log 2>&1
echo [%date% %time%] stage3 training exited >> pipeline.log

rem Guard: only eval if training actually produced the adapter.
if not exist train\adapters\eyoun-s3\adapter_model.safetensors (
  echo [%date% %time%] stage3 pipeline ABORTED - no eyoun-s3 adapter, training failed >> pipeline.log
  exit /b 1
)

rem --- 4) eval eyoun-s3 with SAME caps as baseline + ANTI-LOOP decode (Phase A win)
rem        + HI-RES eval to match training. NOTE: eyoun-s3 was trained on the MERGED
rem        eyoun-s2 base, so eval loads that base + the eyoun-s3 adapter (NOT base_model).
rem        Small line images are unaffected by the hi-res cap; only docs benefit. ---
set MB=train\merged\eyoun-s2-merged-16bit
set DEC=--repetition_penalty 1.2 --no_repeat_ngram_size 6 --max_pixels 1003520
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s3 --name eyoun-s3 --suite arocrbench_arabicocr --limit 50 %DEC% >> eval\eyoun_s3_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s3 --name eyoun-s3 --suite arocrbench_hindawi,arocrbench_historyar,arocrbench_khattparagraph --limit 200 %DEC% >> eval\eyoun_s3_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s3 --name eyoun-s3 --suite arocrbench_patsocr,arocrbench_isippt,arocrbench_synthesizear --limit 500 %DEC% >> eval\eyoun_s3_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s3 --name eyoun-s3 --suite nakba_test --limit 300 %DEC% >> eval\eyoun_s3_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s3 --name eyoun-s3 --suite sedra_handwritten,misraj_dococr %DEC% >> eval\eyoun_s3_run.log 2>&1
echo [%date% %time%] stage3 pipeline COMPLETE (eyoun-s3 evaled) >> pipeline.log
