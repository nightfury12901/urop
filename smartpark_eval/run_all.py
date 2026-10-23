"""
Master runner: trains all protocols × models × modes × 5 seeds,
then runs evaluation, quantization, ablation, and report.
Run from repo root: python smartpark_eval/run_all.py
"""
import subprocess
import sys
import os
import json
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

PROTOCOLS = ["protocol_b", "protocol_a", "protocol_b_author", "protocol_c1", "protocol_c2", "protocol_d"]
MODELS = ["M0", "M1"]
MODES = ["clean", "degraded"]
SEEDS = [0, 1, 2, 3, 4]

def run(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"Running: {desc or cmd}")
    print(f"{'='*60}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"WARNING: command exited with code {ret.returncode}: {cmd}")
    return ret.returncode

def all_trained(protocol, model, mode):
    for s in SEEDS:
        p = f"smartpark_eval/results/models/{protocol}_{model}_{mode}_seed{s}.keras"
        if not os.path.exists(p):
            return False
    return True

def main():
    # Step 1: Training
    for protocol in PROTOCOLS:
        csv = f"smartpark_eval/splits/{protocol}.csv"
        if not os.path.exists(csv):
            print(f"Skipping {protocol} — no split CSV found")
            continue
        for model in MODELS:
            for mode in MODES:
                if all_trained(protocol, model, mode):
                    print(f"Already trained: {protocol} {model} {mode}, skipping.")
                    continue
                # Find which seeds are missing
                missing_seeds = [s for s in SEEDS if not os.path.exists(
                    f"smartpark_eval/results/models/{protocol}_{model}_{mode}_seed{s}.keras")]
                if not missing_seeds:
                    continue
                seeds_str = " ".join(map(str, missing_seeds))
                run(
                    f"python smartpark_eval/src/train.py --protocol {protocol} --model {model} --mode {mode} --seeds {seeds_str}",
                    f"Train {protocol} {model} {mode} seeds {missing_seeds}"
                )

    # Step 2: Evaluation - collect all per-seed results
    all_metrics_rows = []
    for protocol in PROTOCOLS:
        csv = f"smartpark_eval/splits/{protocol}.csv"
        if not os.path.exists(csv):
            continue
        for model in MODELS:
            for mode in MODES:
                for seed in SEEDS:
                    mpath = f"smartpark_eval/results/models/{protocol}_{model}_{mode}_seed{seed}.keras"
                    if not os.path.exists(mpath):
                        continue
                    run(
                        f"python smartpark_eval/src/evaluate.py --protocol {protocol} --model {model} --mode {mode} --seeds {seed}",
                        f"Evaluate {protocol} {model} {mode} seed {seed}"
                    )

    # Step 3: Aggregate metrics_all.csv
    from smartpark_eval.src.evaluate import evaluate_single_model, find_optimal_threshold, bootstrap_group_ci, evaluate_predictions
    import tensorflow as tf
    from smartpark_eval.src.dataset import get_or_create_cached_split

    print("\n\nAggregating metrics across all seeds...")
    all_rows = []
    for protocol in PROTOCOLS:
        csv = f"smartpark_eval/splits/{protocol}.csv"
        if not os.path.exists(csv):
            continue
        df_split = pd.read_csv(csv)
        test_splits = [s for s in df_split["split"].unique() if s.startswith("test")]

        for test_s in test_splits:
            for model in MODELS:
                for mode in MODES:
                    degrade = (mode == "degraded")
                    seed_accs, seed_baccs, seed_precs, seed_recs, seed_f1s, seed_aucs = [], [], [], [], [], []
                    seed_ci_los, seed_ci_his = [], []

                    for seed in SEEDS:
                        mpath = f"smartpark_eval/results/models/{protocol}_{model}_{mode}_seed{seed}.keras"
                        if not os.path.exists(mpath):
                            continue
                        try:
                            res = evaluate_single_model(mpath, csv, test_s, mode, {"paths": {
                                "results_dir": "smartpark_eval/results",
                                "figures_dir": "smartpark_eval/results/figures"
                            }})
                            m = res["metrics_val_tuned"]
                            seed_accs.append(m["accuracy"])
                            seed_baccs.append(m["balanced_accuracy"])
                            seed_precs.append(m["precision"])
                            seed_recs.append(m["recall"])
                            seed_f1s.append(m["f1"])
                            seed_aucs.append(m["roc_auc"])
                            seed_ci_los.append(res["ci_95_lo"])
                            seed_ci_his.append(res["ci_95_hi"])
                        except Exception as e:
                            print(f"Evaluation error for {protocol} {model} {mode} seed {seed}: {e}")

                    if seed_accs:
                        all_rows.append({
                            "protocol": protocol,
                            "test_split": test_s,
                            "model": model,
                            "mode": mode,
                            "n_seeds": len(seed_accs),
                            "acc_mean": round(float(np.mean(seed_accs)), 4),
                            "acc_std": round(float(np.std(seed_accs)), 4),
                            "bacc_mean": round(float(np.mean(seed_baccs)), 4),
                            "bacc_std": round(float(np.std(seed_baccs)), 4),
                            "prec_mean": round(float(np.mean(seed_precs)), 4),
                            "prec_std": round(float(np.std(seed_precs)), 4),
                            "rec_mean": round(float(np.mean(seed_recs)), 4),
                            "rec_std": round(float(np.std(seed_recs)), 4),
                            "f1_mean": round(float(np.mean(seed_f1s)), 4),
                            "f1_std": round(float(np.std(seed_f1s)), 4),
                            "auc_mean": round(float(np.mean(seed_aucs)), 4),
                            "auc_std": round(float(np.std(seed_aucs)), 4),
                            "ci_lo_mean": round(float(np.mean(seed_ci_los)), 4),
                            "ci_hi_mean": round(float(np.mean(seed_ci_his)), 4),
                        })

    if all_rows:
        df_metrics = pd.DataFrame(all_rows)
        df_metrics.to_csv("smartpark_eval/results/metrics_all.csv", index=False)
        print("Saved smartpark_eval/results/metrics_all.csv")
        print(df_metrics.to_string(index=False))

    # Step 4: Quantization (M1 seed 0, protocol_b clean)
    run(
        "python smartpark_eval/src/quantize.py --protocol protocol_b --seed 0 --mode clean",
        "INT8 Quantization: protocol_b M1 clean"
    )

    # Step 5: Ablation
    run(
        "python smartpark_eval/src/ablate.py --protocol protocol_b --mode clean --seed 0",
        "Ablation study"
    )

    # Step 6: Report (figures + LaTeX)
    run("python -m smartpark_eval.src.report", "Generate figures and LaTeX tables")

    print("\n\nAll done! Check smartpark_eval/results/")

if __name__ == "__main__":
    main()
