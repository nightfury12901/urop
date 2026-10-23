import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import yaml

sys.path.insert(0, os.path.abspath("."))
from smartpark_eval.src.dataset import get_or_create_cached_split
from smartpark_eval.src.fallback_hsv_lap import tune_fallback_thresholds, predict_fallback
from smartpark_eval.src.quantize import run_int8_inference

def run_ablation(protocol="protocol_b", mode="clean", seed=0, config_path="smartpark_eval/configs/default.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    splits_dir = cfg["paths"]["splits_dir"]
    results_dir = cfg["paths"]["results_dir"]
    models_dir = os.path.join(results_dir, "models")
    protocol_csv = os.path.join(splits_dir, f"{protocol}.csv")

    degrade = (mode == "degraded")

    # 1. Load validation data for tuning
    print(f"Loading validation set for {protocol} ({mode})...")
    x_val_u8, y_val, _ = get_or_create_cached_split(protocol_csv, "val", degrade=degrade)

    # 2. Load test data
    df_split = pd.read_csv(protocol_csv)
    test_splits = [s for s in df_split["split"].unique() if s.startswith("test")]
    test_s = test_splits[0]
    print(f"Loading test set '{test_s}' for {protocol} ({mode})...")
    x_test_u8, y_test, _ = get_or_create_cached_split(protocol_csv, test_s, degrade=degrade)

    ablation_rows = []

    # --- Method 1: Fallback Alone ---
    print("\n--- Tuning and Evaluating Fallback (HSV + Laplacian) ---")
    val_images_f = x_val_u8.astype(np.float32) / 255.0
    fallback_params = tune_fallback_thresholds(val_images_f, y_val)
    
    test_images_f = x_test_u8.astype(np.float32) / 255.0
    fb_preds, fb_scores = predict_fallback(test_images_f, fallback_params)

    ablation_rows.append({
        "method": "Fallback Alone (HSV Sat/Val + Laplacian)",
        "protocol": protocol,
        "test_set": test_s,
        "mode": mode,
        "accuracy": round(float(accuracy_score(y_test, fb_preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, fb_preds)), 4),
        "precision": round(float(precision_score(y_test, fb_preds, pos_label=1, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, fb_preds, pos_label=1, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, fb_preds, pos_label=1, zero_division=0)), 4),
        "notes": f"Tuned on Val: tau_L={fallback_params['tau_L']:.1f}, tau_S={fallback_params['tau_S']:.2f}, tau_V={fallback_params['tau_V']:.2f}"
    })

    # --- Method 2: M0 (Paper Baseline FP32) ---
    m0_path = os.path.join(models_dir, f"{protocol}_M0_{mode}_seed{seed}.keras")
    if os.path.exists(m0_path):
        print(f"\n--- Evaluating M0 FP32 Baseline ({m0_path}) ---")
        m0 = tf.keras.models.load_model(m0_path)
        m0_probs = m0.predict(test_images_f, batch_size=128, verbose=0).flatten()
        m0_preds = (m0_probs >= 0.5).astype(int)
        ablation_rows.append({
            "method": "M0 Baseline (FP32 CNN, Flatten+Dense)",
            "protocol": protocol,
            "test_set": test_s,
            "mode": mode,
            "accuracy": round(float(accuracy_score(y_test, m0_preds)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, m0_preds)), 4),
            "precision": round(float(precision_score(y_test, m0_preds, pos_label=1, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, m0_preds, pos_label=1, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, m0_preds, pos_label=1, zero_division=0)), 4),
            "notes": "285,857 params, 1.09 MB FP32"
        })

    # --- Method 3: M1 (Compact Target FP32) ---
    m1_path = os.path.join(models_dir, f"{protocol}_M1_{mode}_seed{seed}.keras")
    if os.path.exists(m1_path):
        print(f"\n--- Evaluating M1 FP32 Compact ({m1_path}) ---")
        m1 = tf.keras.models.load_model(m1_path)
        m1_probs = m1.predict(test_images_f, batch_size=128, verbose=0).flatten()
        m1_preds = (m1_probs >= 0.5).astype(int)
        ablation_rows.append({
            "method": "M1 Compact (FP32 CNN, GAP+BN)",
            "protocol": protocol,
            "test_set": test_s,
            "mode": mode,
            "accuracy": round(float(accuracy_score(y_test, m1_preds)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, m1_preds)), 4),
            "precision": round(float(precision_score(y_test, m1_preds, pos_label=1, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, m1_preds, pos_label=1, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, m1_preds, pos_label=1, zero_division=0)), 4),
            "notes": "24,097 params, 94.1 KB FP32"
        })

    # --- Method 4: M1 INT8 (Quantized) ---
    m1_int8_path = os.path.join(models_dir, f"{protocol}_M1_{mode}_int8.tflite")
    if os.path.exists(m1_int8_path):
        print(f"\n--- Evaluating M1 INT8 TFLite ({m1_int8_path}) ---")
        with open(m1_int8_path, "rb") as f:
            int8_bytes = f.read()
        int8_probs, _ = run_int8_inference(int8_bytes, x_test_u8)
        int8_preds = (int8_probs >= 0.5).astype(int)
        ablation_rows.append({
            "method": "M1 Compact (TFLite Full-INT8)",
            "protocol": protocol,
            "test_set": test_s,
            "mode": mode,
            "accuracy": round(float(accuracy_score(y_test, int8_preds)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, int8_preds)), 4),
            "precision": round(float(precision_score(y_test, int8_preds, pos_label=1, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, int8_preds, pos_label=1, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, int8_preds, pos_label=1, zero_division=0)), 4),
            "notes": f"Full-integer INT8: {len(int8_bytes)/1024:.2f} KB"
        })

    # Optional M2 Evidential note
    ablation_rows.append({
        "method": "M1 + Evidential Abstention (M2)",
        "protocol": protocol,
        "test_set": test_s,
        "mode": mode,
        "accuracy": "unvalidated",
        "balanced_accuracy": "unvalidated",
        "precision": "unvalidated",
        "recall": "unvalidated",
        "f1": "unvalidated",
        "notes": "Unvalidated (M2 evidential head skipped; see CLAIMS_TO_UPDATE.md)"
    })

    df_ablation = pd.DataFrame(ablation_rows)
    ablation_csv_path = os.path.join(results_dir, "ablation.csv")
    df_ablation.to_csv(ablation_csv_path, index=False)
    print(f"\nSaved ablation results to {ablation_csv_path}")
    print(df_ablation.to_string(index=False))
    return df_ablation

def main():
    parser = argparse.ArgumentParser(description="Run ablation study.")
    parser.add_argument("--config", default="smartpark_eval/configs/default.yaml")
    parser.add_argument("--protocol", default="protocol_b")
    parser.add_argument("--mode", default="clean")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_ablation(protocol=args.protocol, mode=args.mode, seed=args.seed, config_path=args.config)

if __name__ == "__main__":
    main()
