# Dev Iteration Program

Get the full pipeline working end-to-end in dev mode within a **5-minute wall-clock budget**.

Read `instructions.md` for overall goals of the project.

## The pipeline

Six stages, run sequentially by `./scripts/run_dev.sh`:

1. `src.data.prepare` — download MovieLens-25M, filter to 2000 movies, train/test split
2. `src.data.generate_queries` — synthetic search queries per movie (template backend)
3. `src.embeddings` — compute embeddings for movies and queries
4. `src.tokenizers` — assign semantic IDs via RQ-kmeans
5. `src.generative` — train generative model (query → semantic ID)
6. `src.evaluation` — compute Recall@30, print results table

## The loop

LOOP UNTIL SATISFIED:

1. Run: `./scripts/run_dev.sh`
2. All output goes to `outputs/dev_run.log`. Check results: `tail -50 outputs/dev_run.log`
3. If the run **crashed** (exit code 1): read the last 50 lines of the log, find the traceback, fix the bug.
4. If the run **timed out** (exit code 124): something is too slow. Simplify: fewer epochs, smaller model, skip expensive steps.
5. If the run **succeeded** (exit code 0): check that Stage 6 printed metrics. If so, you're done.
6. Commit your fix, go to step 1.

**Crashes**: If it's a typo or missing import, fix and re-run. If you can't fix it after 3 attempts, simplify the approach.

**NEVER STOP**: Do not pause to ask the human. Keep iterating until the pipeline passes, and all the requirements are satisfied or you are manually stopped.

## What you can modify

- All files under `src/` — implement stages, tune parameters
- `configs/config.py` — adjust dev preset if things are too slow
- `scripts/run_dev.sh` — only if the runner itself has a bug

## What you cannot modify

- `program_dev.md` (this file)
- The 5-minute timeout

## What you should log

The lessons and design principles you learn along the way should be logged in lessons.md as you go along.

## Tips for fitting in 5 minutes

- Stage 1 downloads ~250MB on first run; cached after that.
- Stage 3: use pre-trained embeddings (no fine-tuning) for the first pass.
- Stage 4: RQ-kmeans on 2000 items is fast.
- Stage 5: small model, few epochs. Overfit is fine — prove the pipeline works first.
- Stage 6: Recall@30 is cheap once you have predictions.

