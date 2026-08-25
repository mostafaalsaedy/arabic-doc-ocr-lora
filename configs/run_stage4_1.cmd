@echo off
rem Stage-4.1 pipeline: SECOND epoch of the real-print doc mix (presightai lever,
rem proven in Stage 4: misraj WER 0.642->0.560). Continues eyoun-s4, lr halved to
rem 5e-6, mix seed 43. RELAUNCH after a crash: auto-resumes from stage4_1 ckpt.
cd /d %EYOUN_HOME%
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
echo [%date% %time%] stage4.1 pipeline started >> pipeline.log

.venv\Scripts\python.exe train\stage4_1.py >> train\stage4_1_run.log 2>&1
echo [%date% %time%] stage4.1 training exited >> pipeline.log

if not exist train\adapters\eyoun-s4-1\adapter_model.safetensors (
  echo [%date% %time%] stage4.1 pipeline ABORTED - no eyoun-s4-1 adapter, training failed >> pipeline.log
  exit /b 1
)

rem eval eyoun-s4-1: SAME caps + anti-loop decode + hi-res; misraj + sedra FIRST.
rem eyoun-s4-1 chains from eyoun-s3b/eyoun-s4 whose base is the MERGED eyoun-s2.
set MB=train\merged\eyoun-s2-merged-16bit
set DEC=--repetition_penalty 1.2 --no_repeat_ngram_size 6 --max_pixels 1003520
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s4-1 --name eyoun-s4-1 --suite misraj_dococr %DEC% >> eval\eyoun_s4_1_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s4-1 --name eyoun-s4-1 --suite sedra_handwritten %DEC% >> eval\eyoun_s4_1_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s4-1 --name eyoun-s4-1 --suite arocrbench_arabicocr --limit 50 %DEC% >> eval\eyoun_s4_1_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s4-1 --name eyoun-s4-1 --suite arocrbench_hindawi,arocrbench_historyar,arocrbench_khattparagraph --limit 200 %DEC% >> eval\eyoun_s4_1_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s4-1 --name eyoun-s4-1 --suite arocrbench_patsocr,arocrbench_isippt,arocrbench_synthesizear --limit 500 %DEC% >> eval\eyoun_s4_1_run.log 2>&1
.venv\Scripts\python.exe eval\run_eval.py --model %MB% --adapter train\adapters\eyoun-s4-1 --name eyoun-s4-1 --suite nakba_test --limit 300 %DEC% >> eval\eyoun_s4_1_run.log 2>&1
echo [%date% %time%] stage4.1 pipeline COMPLETE (eyoun-s4-1 evaled) >> pipeline.log
