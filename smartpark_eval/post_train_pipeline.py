"""
post_train_pipeline.py
======================
Runs everything AFTER training is done.
Run from repo root:
    python smartpark_eval/post_train_pipeline.py

Steps:
1. Evaluate all trained models, aggregate metrics_all.csv
2. Run quantize.py on best protocol_b M1 seed
3. Run fallback_hsv_lap.py
4. Run ablation comparison
5. Run report.py to produce figures + LaTeX
6. Write RESULTS.md and CLAIMS_TO_UPDATE.md
"""
import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))

RESULTS_DIR    = "smartpark_eval/results"
SPLITS_DIR     = "smartpark_eval/splits"
FIGURES_DIR    = "smartpark_eval/results/figures"
LATEX_DIR      = "smartpark_eval/results/latex"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LATEX_DIR, exist_ok=True)

PROTOCOLS = ["protocol_b", "protocol_a", "protocol_b_author", "protocol_c1", "protocol_c2", "protocol_d"]
MODELS    = ["M0", "M1"]
MODES     = ["clean", "degraded"]
SEEDS     = [0, 1, 2, 3, 4]

def run(cmd):
    print(f"\n>>> {cmd}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"[WARNING] command returned code {r.returncode}")
    return r.returncode

# ------------------------------------------------------------------ #
# Step 1: Evaluate every trained model
# ------------------------------------------------------------------ #
def step_evaluate():
    from smartpark_eval.src.evaluate import evaluate_single_model
    all_rows = []
    for protocol in PROTOCOLS:
        csv = f"{SPLITS_DIR}/{protocol}.csv"
        if not os.path.exists(csv):
            print(f"Skipping {protocol} — no split CSV")
            continue
        df_s = pd.read_csv(csv)
        test_splits = [s for s in df_s["split"].unique() if s.startswith("test")]

        for model in MODELS:
            for mode in MODES:
                seed_rows = []
                for seed in SEEDS:
                    mpath = f"{RESULTS_DIR}/models/{protocol}_{model}_{mode}_seed{seed}.keras"
                    if not os.path.exists(mpath):
                        continue
                    print(f"  Evaluating {protocol} {model} {mode} seed {seed}...")
                    try:
                        for ts in test_splits:
                            res = evaluate_single_model(mpath, csv, ts, mode, {
                                "paths": {
                                    "results_dir": RESULTS_DIR,
                                    "figures_dir": FIGURES_DIR
                                }
                            })
                            m = res["metrics_val_tuned"]
                            seed_rows.append({
                                "protocol": protocol, "test_split": ts,
                                "model": model, "mode": mode, "seed": seed,
                                "acc": round(m["accuracy"], 4),
                                "bacc": round(m["balanced_accuracy"], 4),
                                "prec": round(m["precision"], 4),
                                "rec": round(m["recall"], 4),
                                "f1": round(m["f1"], 4),
                                "auc": round(m["roc_auc"], 4),
                                "ci_lo": round(res["ci_95_lo"], 4),
                                "ci_hi": round(res["ci_95_hi"], 4),
                                "threshold": round(res["val_tuned_threshold"], 4),
                            })
                    except Exception as e:
                        print(f"  ERROR: {e}")

                if seed_rows:
                    all_rows.extend(seed_rows)

    if not all_rows:
        print("No evaluation results found. Make sure models are trained.")
        return pd.DataFrame()

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(f"{RESULTS_DIR}/metrics_all.csv", index=False)
    print(f"\nSaved {RESULTS_DIR}/metrics_all.csv with {len(df_all)} rows")
    print(df_all.groupby(["protocol", "model", "mode"])[["acc", "f1", "auc"]].mean().to_string())
    return df_all

# ------------------------------------------------------------------ #
# Step 2: Quantize M1 on protocol_b clean seed 0
# ------------------------------------------------------------------ #
def step_quantize():
    run("python smartpark_eval/src/quantize.py --protocol protocol_b --seed 0 --mode clean")

# ------------------------------------------------------------------ #
# Step 3: Fallback
# ------------------------------------------------------------------ #
def step_fallback():
    run("python smartpark_eval/src/fallback_hsv_lap.py --protocol protocol_b --mode clean")

# ------------------------------------------------------------------ #
# Step 4: Ablation
# ------------------------------------------------------------------ #
def step_ablate():
    run("python smartpark_eval/src/ablate.py --protocol protocol_b --mode clean --seed 0")

# ------------------------------------------------------------------ #
# Step 5: Report (figures + LaTeX)
# ------------------------------------------------------------------ #
def step_report():
    run("python smartpark_eval/src/report.py")

# ------------------------------------------------------------------ #
# Step 6: Write training curves from saved histories
# ------------------------------------------------------------------ #
def step_learning_curves():
    hist_dir = f"{RESULTS_DIR}/histories"
    if not os.path.exists(hist_dir):
        return
    files = [f for f in os.listdir(hist_dir) if f.endswith(".json")]
    if not files:
        return

    import matplotlib.ticker as mticker
    from matplotlib import rcParams
    rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})

    for fname in files:
        tag = fname.replace(".json", "")
        with open(os.path.join(hist_dir, fname)) as fh:
            hist = json.load(fh)

        epochs = range(1, len(hist["loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))

        ax1.plot(epochs, hist["loss"], label="Train", linewidth=1.2)
        ax1.plot(epochs, hist["val_loss"], label="Val", linewidth=1.2)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title(f"Loss ({tag})")
        ax1.legend(fontsize=8); ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        ax2.plot(epochs, hist["accuracy"], label="Train", linewidth=1.2)
        ax2.plot(epochs, hist["val_accuracy"], label="Val", linewidth=1.2)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        ax2.set_title(f"Accuracy ({tag})")
        ax2.legend(fontsize=8); ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, f"training_curve_{tag}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved training curve: {save_path}")

# ------------------------------------------------------------------ #
# Step 7: Write RESULTS.md and CLAIMS_TO_UPDATE.md
# ------------------------------------------------------------------ #
def step_write_reports(df_all):
    quant_json = f"{RESULTS_DIR}/quantization.json"
    ablate_json = f"{RESULTS_DIR}/ablation.json"

    quant_data = {}
    if os.path.exists(quant_json):
        with open(quant_json) as f:
            quant_data = json.load(f)

    ablate_data = {}
    if os.path.exists(ablate_json):
        with open(ablate_json) as f:
            ablate_data = json.load(f)

    # --- RESULTS.md ---
    lines = ["# RESULTS.md\n",
             "Auto-generated from post_train_pipeline.py. All numbers produced by code.\n\n"]

    if not df_all.empty:
        lines.append("## Aggregate Performance (mean across 5 seeds)\n\n")
        agg = df_all.groupby(["protocol", "model", "mode"]).agg(
            n_seeds=("seed", "count"),
            acc_mean=("acc", "mean"), acc_std=("acc", "std"),
            f1_mean=("f1", "mean"), f1_std=("f1", "std"),
            auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ).reset_index()
        lines.append(agg.to_markdown(index=False, floatfmt=".4f"))
        lines.append("\n\n")

        # Best result
        if not agg.empty:
            best = agg.loc[agg["acc_mean"].idxmax()]
            lines.append(f"**Best configuration**: {best['protocol']} | {best['model']} | {best['mode']}\n")
            lines.append(f"- Accuracy: {best['acc_mean']:.4f} ± {best['acc_std']:.4f}\n")
            lines.append(f"- F1:       {best['f1_mean']:.4f} ± {best['f1_std']:.4f}\n")
            lines.append(f"- AUC:      {best['auc_mean']:.4f} ± {best['auc_std']:.4f}\n\n")

    if quant_data:
        lines.append("## Quantization Results\n\n")
        for k, v in quant_data.items():
            if isinstance(v, dict):
                lines.append(f"### {k}\n")
                for kk, vv in v.items():
                    lines.append(f"- **{kk}**: {vv}\n")
            else:
                lines.append(f"- **{k}**: {v}\n")
        lines.append("\n")

    if ablate_data:
        lines.append("## Ablation Study\n\n")
        try:
            df_ab = pd.DataFrame(ablate_data)
            lines.append(df_ab.to_markdown(index=False, floatfmt=".4f"))
            lines.append("\n\n")
        except Exception:
            lines.append(str(ablate_data) + "\n\n")

    with open(f"{RESULTS_DIR}/../RESULTS.md", "w") as f:
        f.writelines(lines)
    print(f"Saved RESULTS.md")

    # --- CLAIMS_TO_UPDATE.md ---
    claim_lines = [
        "# CLAIMS_TO_UPDATE.md\n\n",
        "This file lists claims in the paper that should be updated based on the reproducible evaluation.\n\n",
        "## Architecture Claims\n\n",
        "| Claim in paper | Reality | Action |\n",
        "|---|---|---|\n",
        "| `8×8×64 = 2,304` (flatten output) | `8×8×64 = 4,096` — arithmetic error | Correct to 4,096 |\n",
        "| Param count of 24,577 | See results/model_M1.md for actual count | Replace with measured count |\n",
        "| FP32 model size 365.8 KB | See quantization.json | Replace with measured size |\n\n",
        "## Evaluation Protocol Claims\n\n",
        "| Claim | Issue | Recommended fix |\n",
        "|---|---|---|\n",
        "| 97.8% accuracy | Obtained with undisclosed split; not reproducible | Report measured accuracy from Protocol B |\n",
        "| Confusion matrix (15 samples) | 15 samples is implausibly small for a 2,400-image claim | Replace with full-dataset confusion matrix |\n",
        "| Validation accuracy flat at 0.57 | Evidence of training failure / severe overfitting | Report training curves from this evaluation |\n\n",
        "## Recommended Updated Numbers\n\n",
        "*(Filled in automatically when evaluation produces results)*\n\n",
    ]

    if not df_all.empty:
        agg = df_all.groupby(["protocol", "model", "mode"]).agg(
            acc_mean=("acc", "mean"), acc_std=("acc", "std"),
            f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        ).reset_index()
        # protocol_b, M1, clean is the primary comparison
        row = agg[(agg["protocol"] == "protocol_b") & (agg["model"] == "M1") & (agg["mode"] == "clean")]
        if not row.empty:
            r = row.iloc[0]
            claim_lines += [
                f"**M1 (24k params), Protocol B (camera-disjoint split), Clean images** (5-seed mean ± std):\n\n",
                f"- Accuracy: {r['acc_mean']:.4f} ± {r['acc_std']:.4f}\n",
                f"- F1:       {r['f1_mean']:.4f} ± {r['f1_std']:.4f}\n\n",
                "This should replace the 97.8% claim, with the updated protocol noted.\n"
            ]

    with open(f"{RESULTS_DIR}/../CLAIMS_TO_UPDATE.md", "w") as f:
        f.writelines(claim_lines)
    print("Saved CLAIMS_TO_UPDATE.md")


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("=" * 60)
    print("SmartPark Post-Training Pipeline")
    print("=" * 60)

    print("\n[1/7] Evaluating all trained models...")
    df_all = step_evaluate()

    print("\n[2/7] Quantizing M1 (INT8)...")
    step_quantize()

    print("\n[3/7] HSV/Laplacian fallback heuristic...")
    step_fallback()

    print("\n[4/7] Ablation study...")
    step_ablate()

    print("\n[5/7] Generating figures and LaTeX tables...")
    step_report()

    print("\n[6/7] Generating training curves from saved histories...")
    step_learning_curves()

    print("\n[7/7] Writing RESULTS.md and CLAIMS_TO_UPDATE.md...")
    step_write_reports(df_all)

    print("\n\n" + "=" * 60)
    print("Pipeline complete. Output files:")
    print(f"  {RESULTS_DIR}/metrics_all.csv")
    print(f"  {RESULTS_DIR}/quantization.json")
    print(f"  {RESULTS_DIR}/ablation.json")
    print(f"  {FIGURES_DIR}/*.pdf  (training curves, confusion matrices, etc.)")
    print(f"  {LATEX_DIR}/*.tex  (LaTeX tables)")
    print(f"  smartpark_eval/RESULTS.md")
    print(f"  smartpark_eval/CLAIMS_TO_UPDATE.md")
    print("=" * 60)
