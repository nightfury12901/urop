"""
quick_eval.py — evaluates any trained models found, produces metrics_all.csv.
Run after at least one seed finishes. Can be re-run to add more results.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

sys.path.insert(0, os.path.abspath("."))

RESULTS_DIR = "smartpark_eval/results"
SPLITS_DIR  = "smartpark_eval/splits"
FIGURES_DIR = "smartpark_eval/results/figures"
LATEX_DIR   = "smartpark_eval/results/latex"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LATEX_DIR,   exist_ok=True)

PROTOCOLS = ["protocol_b", "protocol_b_author", "protocol_a",
             "protocol_c1", "protocol_c2", "protocol_d"]
MODELS    = ["M0", "M1"]
MODES     = ["clean", "degraded"]
SEEDS     = [0, 1, 2, 3, 4]

# ------------------------------------------------------------------ #
def load_split_data(protocol_csv, split_name, mode):
    """Load images + labels directly (no cache required, uses cache if present)."""
    from smartpark_eval.src.dataset import get_or_create_cached_split
    x, y, meta = get_or_create_cached_split(
        protocol_csv, split_name, degrade=(mode == "degraded")
    )
    # npz stores dict as 0-d object array; unwrap it
    if isinstance(meta, np.ndarray) and meta.ndim == 0:
        meta = meta.item()
    return x, y, meta

def find_threshold(val_probs, val_labels):
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        s = balanced_accuracy_score(val_labels, (val_probs >= t).astype(int))
        if s > best_s:
            best_s = s
            best_t = t
    return float(best_t)

def metrics(y_true, y_probs, t=0.5):
    p = (y_probs >= t).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5
    return dict(
        accuracy   = round(float(accuracy_score(y_true, p)), 4),
        bacc       = round(float(balanced_accuracy_score(y_true, p)), 4),
        precision  = round(float(precision_score(y_true, p, zero_division=0)), 4),
        recall     = round(float(recall_score(y_true, p, zero_division=0)), 4),
        f1         = round(float(f1_score(y_true, p, zero_division=0)), 4),
        auc        = round(auc, 4),
        threshold  = round(t, 4),
        n_samples  = int(len(y_true)),
    )

def bootstrap_ci(y_true, y_probs, t, meta, n=500, seed=42):
    rng = np.random.default_rng(seed)
    # meta may be a dict (from npz) or already a DataFrame
    if isinstance(meta, dict):
        df = pd.DataFrame(meta)
    elif isinstance(meta, pd.DataFrame):
        df = meta
    else:
        # fallback: treat every sample as its own group
        df = pd.DataFrame({"gk": np.arange(len(y_true)).astype(str)})
    if "camera" in df.columns and "date" in df.columns:
        df["gk"] = df["camera"].astype(str) + "_" + df["date"].astype(str)
    elif "gk" not in df.columns:
        df["gk"] = np.arange(len(df)).astype(str)
    groups = df["gk"].unique()
    if len(groups) <= 1:
        acc = accuracy_score(y_true, (y_probs >= t).astype(int))
        return acc, acc
    g2i = {g: np.where(df["gk"].values == g)[0] for g in groups}
    boot = []
    for _ in range(n):
        sg = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([g2i[g] for g in sg])
        boot.append(accuracy_score(y_true[idx], (y_probs[idx] >= t).astype(int)))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def eval_model(mpath, csv, test_split, mode):
    import tensorflow as tf
    model = tf.keras.models.load_model(mpath)

    x_val, y_val, _ = load_split_data(csv, "val", mode)
    x_test, y_test, meta = load_split_data(csv, test_split, mode)

    vp = model.predict(x_val.astype(np.float32)/255.0, batch_size=256, verbose=0).flatten()
    tp = model.predict(x_test.astype(np.float32)/255.0, batch_size=256, verbose=0).flatten()

    t  = find_threshold(vp, y_val)
    m  = metrics(y_test, tp, t)
    ci_lo, ci_hi = bootstrap_ci(y_test, tp, t, meta, n=500)
    m["ci_lo"] = round(ci_lo, 4)
    m["ci_hi"] = round(ci_hi, 4)
    return m, tp, y_test, x_test

# ------------------------------------------------------------------ #
# Confusion matrix figure
def save_cm(y_true, y_probs, t, save_path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"font.family": "serif", "font.size": 9})

    cm = confusion_matrix(y_true, (y_probs >= t).astype(int), labels=[0, 1])
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(cm_n, vmin=0, vmax=1, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({cm_n[i,j]:.2%})",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_n[i,j] > 0.5 else "black")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Free","Occ"]); ax.set_yticklabels(["Free","Occ"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------------ #
# Training curves
def save_curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib import rcParams
    rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})

    hist_dir = f"{RESULTS_DIR}/histories"
    if not os.path.exists(hist_dir):
        return
    for fname in os.listdir(hist_dir):
        if not fname.endswith(".json"):
            continue
        tag = fname.replace(".json", "")
        with open(os.path.join(hist_dir, fname)) as f:
            h = json.load(f)
        epochs = range(1, len(h["loss"]) + 1)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
        a1.plot(epochs, h["loss"], label="Train", lw=1.2)
        a1.plot(epochs, h["val_loss"], label="Val", lw=1.2, linestyle="--")
        a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.legend(fontsize=8)
        a1.set_title(f"Loss · {tag}", fontsize=8)
        a1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        a2.plot(epochs, h["accuracy"], label="Train", lw=1.2)
        a2.plot(epochs, h["val_accuracy"], label="Val", lw=1.2, linestyle="--")
        a2.set_xlabel("Epoch"); a2.set_ylabel("Accuracy"); a2.legend(fontsize=8)
        a2.set_title(f"Accuracy · {tag}", fontsize=8)
        a2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.tight_layout()
        sp = os.path.join(FIGURES_DIR, f"curve_{tag}.pdf")
        plt.savefig(sp, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved {sp}")

# ------------------------------------------------------------------ #
def latex_table(df_agg):
    """Produce a LaTeX booktabs table from aggregated results."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{SmartPark CNN Evaluation Results (mean $\pm$ std, 5 seeds)}",
        r"\label{tab:results}",
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Protocol & Model & Mode & Test & Acc & Bal-Acc & F1 & AUC \\",
        r"\midrule",
    ]
    for _, r in df_agg.iterrows():
        def fmt(col):
            mean_val = r.get(col, float('nan'))
            std_col = col.replace('_mean', '_std')
            std_val = r.get(std_col, float('nan'))
            if pd.isna(std_val):
                return f"${mean_val:.4f}$"
            return f"${mean_val:.4f}\\pm{std_val:.4f}$"
        lines.append(
            f"{r['protocol']} & {r['model']} & {r['mode']} & {r.get('test_split','test')} & "
            f"{fmt('accuracy_mean')} & "
            f"{fmt('bacc_mean')} & "
            f"{fmt('f1_mean')} & "
            f"{fmt('auc_mean')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
def main():
    print("=" * 65)
    print("SmartPark Quick Evaluation")
    print("=" * 65)

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
                for seed in SEEDS:
                    mpath = f"{RESULTS_DIR}/models/{protocol}_{model}_{mode}_seed{seed}.keras"
                    if not os.path.exists(mpath):
                        continue
                    print(f"Evaluating {protocol} | {model} | {mode} | seed {seed}...")
                    for ts in test_splits:
                        try:
                            m, tp, yt, xt = eval_model(mpath, csv, ts, mode)
                            print(f"  {ts}: acc={m['accuracy']:.4f} bacc={m['bacc']:.4f} "
                                  f"f1={m['f1']:.4f} auc={m['auc']:.4f} "
                                  f"CI=[{m['ci_lo']:.4f},{m['ci_hi']:.4f}]")
                            # Save confusion matrix for seed 0
                            if seed == 0:
                                cm_path = f"{FIGURES_DIR}/cm_{protocol}_{model}_{mode}_{ts}.pdf"
                                save_cm(yt, tp, m["threshold"], cm_path,
                                        title=f"{protocol} | {model} | {mode} | {ts}")
                            all_rows.append({"protocol": protocol, "model": model,
                                             "mode": mode, "test_split": ts, "seed": seed, **m})
                        except Exception as e:
                            print(f"  ERROR: {e}")

    if not all_rows:
        print("\nNo models found yet. Try again after training completes.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{RESULTS_DIR}/metrics_all.csv", index=False)
    print(f"\nSaved {RESULTS_DIR}/metrics_all.csv ({len(df)} rows)")

    # Aggregate
    agg_cols = ["accuracy", "bacc", "precision", "recall", "f1", "auc"]
    agg = df.groupby(["protocol", "model", "mode", "test_split"]).agg(
        n_seeds=("seed", "count"),
        **{f"{c}_mean": (c, "mean") for c in agg_cols},
        **{f"{c}_std":  (c, "std")  for c in agg_cols},
        ci_lo=("ci_lo", "mean"),
        ci_hi=("ci_hi", "mean"),
    ).reset_index()
    agg.to_csv(f"{RESULTS_DIR}/metrics_agg.csv", index=False)
    print(f"Saved {RESULTS_DIR}/metrics_agg.csv")
    print("\n" + agg[["protocol","model","mode","test_split","n_seeds","accuracy_mean","accuracy_std","f1_mean","auc_mean"]].to_string(index=False))

    # LaTeX
    latex = latex_table(agg)
    with open(f"{LATEX_DIR}/results_table.tex", "w") as f:
        f.write(latex)
    print(f"\nSaved {LATEX_DIR}/results_table.tex")

    # Training curves
    print("\nGenerating training curves...")
    save_curves()

    # RESULTS.md (tabulate-free markdown table builder)
    def df_to_md(df_in):
        cols = df_in.columns.tolist()
        hdr  = "| " + " | ".join(str(c) for c in cols) + " |"
        sep  = "| " + " | ".join("---" for _ in cols) + " |"
        rows = []
        for _, row in df_in.iterrows():
            def fv(v):
                if isinstance(v, float): return f"{v:.4f}" if not pd.isna(v) else ""
                return str(v)
            rows.append("| " + " | ".join(fv(row[c]) for c in cols) + " |")
        return "\n".join([hdr, sep] + rows)

    lines = ["# RESULTS.md\n\n",
             "_All numbers produced by code. Zero manual edits._\n\n",
             "## Per-seed results\n\n",
             df_to_md(df), "\n\n",
             "## Aggregated (mean \u00b1 std across seeds)\n\n",
             df_to_md(agg), "\n"]
    with open(f"smartpark_eval/RESULTS.md", "w") as f:
        f.writelines(lines)
    print("Saved smartpark_eval/RESULTS.md")


    # CLAIMS_TO_UPDATE.md
    best = agg.sort_values("accuracy_mean", ascending=False).iloc[0]
    p_b_m1 = agg[(agg.protocol=="protocol_b")&(agg.model=="M1")&(agg.mode=="clean")]
    claim_lines = [
        "# CLAIMS_TO_UPDATE.md\n\n",
        "_Produced by quick_eval.py — all numbers from code._\n\n",
        "## Architecture\n\n",
        "| Paper claim | Correct value |\n|---|---|\n",
        "| 8×8×64 flattens to 2,304 | 8×8×64 = **4,096** (arithmetic error) |\n",
        "| 24,577 parameters (M1) | See `results/model_M1.md` |\n",
        "| 365.8 KB FP32 size | See `results/quantization.json` |\n\n",
        "## Evaluation\n\n",
        "| Paper claim | Honest result |\n|---|---|\n",
        "| 97.8% accuracy on 2,400 patches | Undisclosed split, cannot reproduce |\n",
        "| Confusion matrix: 15 samples | Implausibly small for 2,400-image claim |\n",
    ]
    if not p_b_m1.empty:
        r = p_b_m1.iloc[0]
        claim_lines += [
            f"| — | Protocol B (camera-disjoint): **acc = {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}** (M1 clean, 5 seeds) |\n",
            f"| — | 95% bootstrap CI: [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] |\n",
        ]
    claim_lines += [
        "\n## Recommended text update\n\n",
        "Replace the 97.8% claim with the honest measured accuracy above, with the protocol clearly stated.\n",
    ]
    with open("smartpark_eval/CLAIMS_TO_UPDATE.md", "w") as f:
        f.writelines(claim_lines)
    print("Saved smartpark_eval/CLAIMS_TO_UPDATE.md")

    print("\n" + "=" * 65)
    print("Done. All outputs in smartpark_eval/results/")
    print("=" * 65)

if __name__ == "__main__":
    main()
