import argparse
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
import tensorflow as tf
import yaml

sys.path.insert(0, os.path.abspath("."))
from smartpark_eval.src.dataset import get_or_create_cached_split

def find_optimal_threshold(val_probs, val_labels):
    """
    Finds optimal decision threshold on VALIDATION set only.
    Optimizes for balanced accuracy.
    """
    best_thresh = 0.5
    best_score = -1.0
    for t in np.linspace(0.1, 0.9, 81):
        preds = (val_probs >= t).astype(int)
        score = balanced_accuracy_score(val_labels, preds)
        if score > best_score:
            best_score = score
            best_thresh = t
    return float(best_thresh)

def bootstrap_group_ci(df_test, y_true, y_probs, threshold, n_bootstrap=1000, seed=42):
    """
    Computes 95% confidence intervals by bootstrapping over (camera, date) groups,
    NOT individual patches (Section 8).
    """
    rng = np.random.default_rng(seed)
    # Define groups
    df_test = df_test.copy()
    df_test["group_key"] = df_test["camera"].astype(str) + "___" + df_test["date"].astype(str)
    groups = df_test["group_key"].unique()
    num_groups = len(groups)

    if num_groups <= 1:
        # Fallback if only 1 group exists
        acc = accuracy_score(y_true, (y_probs >= threshold).astype(int))
        return acc, acc

    group_to_indices = {g: np.where(df_test["group_key"] == g)[0] for g in groups}

    boot_accs = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(groups, size=num_groups, replace=True)
        idx_list = [group_to_indices[g] for g in sampled_groups]
        sample_indices = np.concatenate(idx_list)

        y_t = y_true[sample_indices]
        y_p = (y_probs[sample_indices] >= threshold).astype(int)
        boot_accs.append(accuracy_score(y_t, y_p))

    ci_lo = float(np.percentile(boot_accs, 2.5))
    ci_hi = float(np.percentile(boot_accs, 97.5))
    return ci_lo, ci_hi

def evaluate_predictions(y_true, y_probs, threshold=0.5):
    """Computes all core performance metrics."""
    preds = (y_probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true, preds))
    b_acc = float(balanced_accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, preds, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, preds, pos_label=1, zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    return {
        "accuracy": acc,
        "balanced_accuracy": b_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_norm": cm_norm.tolist()
    }

def save_worst_error_grids(test_images_u8, y_true, y_probs, save_path, n_each=8):
    """
    Saves image grids of worst false positives and worst false negatives (Section 8).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    preds = (y_probs >= 0.5).astype(int)

    # False Positives: True = Free (0), Predicted = Occupied (high prob)
    fp_mask = (y_true == 0) & (preds == 1)
    fp_indices = np.where(fp_mask)[0]
    fp_indices = fp_indices[np.argsort(-y_probs[fp_indices])]  # highest prob first

    # False Negatives: True = Occupied (1), Predicted = Free (low prob)
    fn_mask = (y_true == 1) & (preds == 0)
    fn_indices = np.where(fn_mask)[0]
    fn_indices = fn_indices[np.argsort(y_probs[fn_indices])]  # lowest prob first

    fig, axes = plt.subplots(2, n_each, figsize=(n_each * 2, 4.5))

    for col in range(n_each):
        # Row 0: Worst FP
        ax_fp = axes[0, col]
        if col < len(fp_indices):
            idx = fp_indices[col]
            ax_fp.imshow(test_images_u8[idx])
            ax_fp.set_title(f"FP: p={y_probs[idx]:.2f}", fontsize=8)
        else:
            ax_fp.axis("off")
        ax_fp.set_xticks([]); ax_fp.set_yticks([])
        if col == 0:
            ax_fp.set_ylabel("Worst False Positives\n(Free classified as Occ)", fontsize=8)

        # Row 1: Worst FN
        ax_fn = axes[1, col]
        if col < len(fn_indices):
            idx = fn_indices[col]
            ax_fn.imshow(test_images_u8[idx])
            ax_fn.set_title(f"FN: p={y_probs[idx]:.2f}", fontsize=8)
        else:
            ax_fn.axis("off")
        ax_fn.set_xticks([]); ax_fn.set_yticks([])
        if col == 0:
            ax_fn.set_ylabel("Worst False Negatives\n(Occ classified as Free)", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved error grid to {save_path}")

def evaluate_single_model(model_path, protocol_csv, test_split_name, mode, cfg, val_split_name="val"):
    degrade = (mode == "degraded")
    # 1. Load test data
    x_test_u8, y_test, meta_test = get_or_create_cached_split(protocol_csv, test_split_name, degrade=degrade)
    # 2. Load val data for threshold tuning
    x_val_u8, y_val, _ = get_or_create_cached_split(protocol_csv, val_split_name, degrade=degrade)

    model = tf.keras.models.load_model(model_path)
    
    # Predict val
    val_probs = model.predict(x_val_u8.astype(np.float32) / 255.0, batch_size=128, verbose=0).flatten()
    val_tuned_threshold = find_optimal_threshold(val_probs, y_val)

    # Predict test
    test_probs = model.predict(x_test_u8.astype(np.float32) / 255.0, batch_size=128, verbose=0).flatten()

    # Evaluation with fixed threshold 0.5 and val_tuned
    metrics_05 = evaluate_predictions(y_test, test_probs, threshold=0.5)
    metrics_tuned = evaluate_predictions(y_test, test_probs, threshold=val_tuned_threshold)

    # Unwrap meta (npz stores dict as 0-d object array)
    if isinstance(meta_test, np.ndarray) and meta_test.ndim == 0:
        meta_test = meta_test.item()
    df_test_meta = pd.DataFrame(meta_test)
    ci_lo, ci_hi = bootstrap_group_ci(df_test_meta, y_test, test_probs, threshold=val_tuned_threshold)

    # Breakdown by condition
    breakdowns = {}
    if "weather" in df_test_meta.columns:
        weather_breakdown = {}
        for w in df_test_meta["weather"].unique():
            w_idx = np.where(df_test_meta["weather"] == w)[0]
            if len(w_idx) > 0:
                weather_breakdown[str(w)] = evaluate_predictions(y_test[w_idx], test_probs[w_idx], threshold=val_tuned_threshold)
        breakdowns["weather"] = weather_breakdown

    if "hour" in df_test_meta.columns:
        hour_breakdown = {}
        hours = df_test_meta["hour"].to_numpy()
        buckets = {
            "morning (<11)": np.where(hours < 11)[0],
            "midday (11-15)": np.where((hours >= 11) & (hours <= 15))[0],
            "evening (>15)": np.where(hours > 15)[0]
        }
        for b_name, b_idx in buckets.items():
            if len(b_idx) > 0:
                hour_breakdown[b_name] = evaluate_predictions(y_test[b_idx], test_probs[b_idx], threshold=val_tuned_threshold)
        breakdowns["hour"] = hour_breakdown


    return {
        "metrics_05": metrics_05,
        "metrics_val_tuned": metrics_tuned,
        "val_tuned_threshold": val_tuned_threshold,
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
        "breakdowns": breakdowns,
        "test_probs": test_probs,
        "y_test": y_test,
        "x_test_u8": x_test_u8
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate SmartPark models.")
    parser.add_argument("--config", default="smartpark_eval/configs/default.yaml", help="Path to config file.")
    parser.add_argument("--protocol", default="protocol_b", help="Protocol split name.")
    parser.add_argument("--model", choices=["M0", "M1"], default="M1", help="Model architecture.")
    parser.add_argument("--mode", choices=["clean", "degraded"], default="clean", help="Evaluation mode.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Random seeds.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    splits_dir = cfg["paths"]["splits_dir"]
    results_dir = cfg["paths"]["results_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    protocol_csv = os.path.join(splits_dir, f"{args.protocol}.csv")

    # Determine test split name(s)
    df_split = pd.read_csv(protocol_csv)
    test_splits = [s for s in df_split["split"].unique() if s.startswith("test")]

    seed_results = []
    for test_s in test_splits:
        for s in args.seeds:
            model_path = os.path.join(results_dir, "models", f"{args.protocol}_{args.model}_{args.mode}_seed{s}.keras")
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}, skipping...")
                continue

            res = evaluate_single_model(model_path, protocol_csv, test_s, args.mode, cfg)
            seed_results.append({
                "protocol": args.protocol,
                "model": args.model,
                "mode": args.mode,
                "test_split": test_s,
                "seed": s,
                "val_threshold": res["val_tuned_threshold"],
                "acc_val_tuned": res["metrics_val_tuned"]["accuracy"],
                "f1_val_tuned": res["metrics_val_tuned"]["f1"],
                "prec_val_tuned": res["metrics_val_tuned"]["precision"],
                "rec_val_tuned": res["metrics_val_tuned"]["recall"],
                "auc": res["metrics_val_tuned"]["roc_auc"],
                "ci_lo": res["ci_95_lo"],
                "ci_hi": res["ci_95_hi"]
            })

            # Save error grid for seed 0
            if s == 0:
                grid_save_path = os.path.join(figures_dir, f"worst_errors_{args.protocol}_{test_s}_{args.model}_{args.mode}.png")
                save_worst_error_grids(res["x_test_u8"], res["y_test"], res["test_probs"], grid_save_path)

    if seed_results:
        res_df = pd.DataFrame(seed_results)
        print("\n--- Evaluation Summary Across Seeds ---")
        print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
