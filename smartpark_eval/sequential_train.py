"""
sequential_train.py
====================
Trains ALL missing models sequentially (one at a time) to avoid OOM.
Run from repo root: python smartpark_eval/sequential_train.py
"""
import os, sys, subprocess, json
sys.path.insert(0, os.path.abspath("."))

RESULTS_DIR = "smartpark_eval/results"
SPLITS_DIR  = "smartpark_eval/splits"

# All tasks in priority order (Protocol B first — primary evaluation)
TASKS = [
    # protocol, model, mode, seeds
    ("protocol_b",        "M1", "clean",    [3, 4]),
    ("protocol_b",        "M0", "clean",    [2, 3, 4]),
    ("protocol_b",        "M1", "degraded", [1, 2, 3, 4]),
    ("protocol_b_author", "M1", "clean",    [0, 1, 2, 3, 4]),
    ("protocol_a",        "M1", "clean",    [0, 1, 2, 3, 4]),
    ("protocol_c1",       "M1", "clean",    [0, 1, 2, 3, 4]),
    ("protocol_c2",       "M1", "clean",    [0, 1, 2, 3, 4]),
    ("protocol_d",        "M1", "clean",    [0, 1, 2, 3, 4]),
]

def model_exists(protocol, model, mode, seed):
    path = f"{RESULTS_DIR}/models/{protocol}_{model}_{mode}_seed{seed}.keras"
    return os.path.exists(path)

def split_exists(protocol):
    return os.path.exists(f"{SPLITS_DIR}/{protocol}.csv")

def run_seed(protocol, model, mode, seed):
    cmd = (f"python smartpark_eval/src/train.py "
           f"--protocol {protocol} --model {model} --mode {mode} --seeds {seed}")
    print(f"\n>>> {cmd}")
    ret = subprocess.run(cmd, shell=True)
    return ret.returncode == 0

def main():
    total_trained = 0
    total_skipped = 0
    total_failed  = 0

    for protocol, model, mode, seeds in TASKS:
        if not split_exists(protocol):
            print(f"[SKIP] {protocol} — no split CSV")
            continue

        for seed in seeds:
            if model_exists(protocol, model, mode, seed):
                print(f"[SKIP] {protocol} {model} {mode} seed {seed} — already trained")
                total_skipped += 1
                continue

            print(f"\n{'='*60}")
            print(f"Training: {protocol} | {model} | {mode} | seed {seed}")
            print(f"{'='*60}")
            ok = run_seed(protocol, model, mode, seed)
            if ok:
                total_trained += 1
                print(f"  ✓ Done: {protocol} {model} {mode} seed {seed}")
            else:
                total_failed += 1
                print(f"  ✗ FAILED: {protocol} {model} {mode} seed {seed}")

    print(f"\n{'='*60}")
    print(f"Sequential training complete.")
    print(f"  Trained: {total_trained}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Failed:  {total_failed}")
    print(f"{'='*60}")

    if total_failed == 0:
        print("\nRunning final evaluation pipeline...")
        subprocess.run("python smartpark_eval/quick_eval.py", shell=True)
        subprocess.run("python smartpark_eval/src/quantize.py --protocol protocol_b --seed 0 --mode clean", shell=True)
        subprocess.run("python smartpark_eval/src/fallback_hsv_lap.py --protocol protocol_b --mode clean", shell=True)
        subprocess.run("python smartpark_eval/src/ablate.py --protocol protocol_b --mode clean --seed 0", shell=True)

if __name__ == "__main__":
    main()
