import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
import yaml

sys.path.insert(0, os.path.abspath("."))
from smartpark_eval.src.dataset import get_or_create_cached_split

def convert_to_tflite_int8(model, representative_data):
    """
    Converts a Keras model to full-integer INT8 TFLite with int8 input and output.
    """
    def representative_dataset():
        for i in range(len(representative_data)):
            yield [np.expand_dims(representative_data[i], axis=0).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    return tflite_model

def convert_to_tflite_fp32(model):
    """Converts a Keras model to baseline FP32 TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()

def estimate_tensor_arena(interpreter):
    """
    Estimates the minimum tensor arena size from tensor details.
    """
    details = interpreter.get_tensor_details()
    max_single_tensor = 0
    for t in details:
        shape = t["shape"]
        dtype = np.dtype(t["dtype"])
        size = int(np.prod(shape) * dtype.itemsize)
        if size > max_single_tensor:
            max_single_tensor = size

    # Heuristic arena estimate for TFLM runtime on ESP32
    arena_estimate = max(max_single_tensor * 3 + 16384, 32768)
    return arena_estimate

def run_int8_inference(tflite_model_bytes, x_u8):
    """
    Runs fast vectorized TFLite interpreter inference, returning float probabilities in [0, 1].
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zero_point = input_details["quantization"]
    out_scale, out_zero_point = output_details["quantization"]
    arena_est = estimate_tensor_arena(interpreter)

    N = len(x_u8)
    x_f = x_u8.astype(np.float32) / 255.0

    if in_scale > 0:
        q_all = np.clip(np.round(x_f / in_scale + in_zero_point), -128, 127).astype(np.int8)
    else:
        q_all = x_u8.view(np.int8)

    probs = np.empty(N, dtype=np.float32)
    in_idx = input_details["index"]
    out_idx = output_details["index"]

    for i in range(N):
        interpreter.set_tensor(in_idx, q_all[i:i+1])
        interpreter.invoke()
        q_out = interpreter.get_tensor(out_idx)
        raw_val = float(np.squeeze(q_out))
        if out_scale > 0:
            prob = (raw_val - float(out_zero_point)) * float(out_scale)
        else:
            prob = raw_val
        probs[i] = np.clip(prob, 0.0, 1.0)

    return probs, arena_est

def find_optimal_threshold(y_true, probs):
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        score = balanced_accuracy_score(y_true, (probs >= t).astype(int))
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t

def compute_metrics_at_threshold(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, pos_label=1, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, pos_label=1, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, pos_label=1, zero_division=0)), 4)
    }

def main():
    parser = argparse.ArgumentParser(description="Quantize M1 models to INT8 and evaluate across seeds.")
    parser.add_argument("--config", default="smartpark_eval/configs/default.yaml", help="Path to config file.")
    parser.add_argument("--protocol", default="protocol_b", help="Protocol name.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Seed models to quantize.")
    parser.add_argument("--mode", default="clean", help="Clean or degraded.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    splits_dir = cfg["paths"]["splits_dir"]
    results_dir = cfg["paths"]["results_dir"]
    models_dir = os.path.join(results_dir, "models")
    protocol_csv = os.path.join(splits_dir, f"{args.protocol}.csv")

    degrade = (args.mode == "degraded")

    # 1. Load representative calibration data (1000 patches)
    print(f"Loading representative calibration data (1,000 patches from train)...")
    x_train_u8, _, _ = get_or_create_cached_split(protocol_csv, "train", degrade=degrade)
    n_rep = min(1000, len(x_train_u8))
    rep_indices = np.linspace(0, len(x_train_u8) - 1, n_rep, dtype=int)
    rep_data = (x_train_u8[rep_indices].astype(np.float32) / 255.0)

    # 2. Load validation and test sets
    print(f"Loading validation split for threshold tuning...")
    x_val_u8, y_val, _ = get_or_create_cached_split(protocol_csv, "val", degrade=degrade)
    x_val_f = x_val_u8.astype(np.float32) / 255.0

    df_split = pd.read_csv(protocol_csv)
    test_splits = [s for s in df_split["split"].unique() if s.startswith("test")]
    test_s = test_splits[0]
    print(f"Loading test split '{test_s}'...")
    x_test_u8, y_test, _ = get_or_create_cached_split(protocol_csv, test_s, degrade=degrade)
    x_test_f = x_test_u8.astype(np.float32) / 255.0

    per_seed_results = []
    fp32_tflite_size = None
    int8_tflite_size = None
    arena_estimate_bytes = None

    for seed in args.seeds:
        model_path = os.path.join(models_dir, f"{args.protocol}_M1_{args.mode}_seed{seed}.keras")
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping seed {seed}")
            continue

        print(f"\n--- Quantizing and Evaluating Seed {seed} ---")
        model = tf.keras.models.load_model(model_path)
        fp32_keras_size = os.path.getsize(model_path)

        # FP32 TFLite
        tflite_fp32 = convert_to_tflite_fp32(model)
        fp32_tflite_path = os.path.join(models_dir, f"{args.protocol}_M1_{args.mode}_seed{seed}_fp32.tflite")
        with open(fp32_tflite_path, "wb") as f:
            f.write(tflite_fp32)
        fp32_tflite_size = len(tflite_fp32)

        # INT8 TFLite
        tflite_int8 = convert_to_tflite_int8(model, rep_data)
        int8_tflite_path = os.path.join(models_dir, f"{args.protocol}_M1_{args.mode}_seed{seed}_int8.tflite")
        with open(int8_tflite_path, "wb") as f:
            f.write(tflite_int8)
        int8_tflite_size = len(tflite_int8)

        # FP32 Val & Test
        val_probs_fp32 = model.predict(x_val_f, batch_size=256, verbose=0).flatten()
        test_probs_fp32 = model.predict(x_test_f, batch_size=256, verbose=0).flatten()
        t_fp32_opt = find_optimal_threshold(y_val, val_probs_fp32)

        fp32_opt_metrics = compute_metrics_at_threshold(y_test, test_probs_fp32, t_fp32_opt)
        fp32_default_metrics = compute_metrics_at_threshold(y_test, test_probs_fp32, 0.50)

        # INT8 Val & Test
        print("Running INT8 inference on validation split for optimal threshold tuning...")
        val_probs_int8, _ = run_int8_inference(tflite_int8, x_val_u8)
        t_int8_opt = find_optimal_threshold(y_val, val_probs_int8)

        print(f"Running INT8 inference on test split ({len(x_test_u8)} samples)...")
        test_probs_int8, arena_est = run_int8_inference(tflite_int8, x_test_u8)
        arena_estimate_bytes = arena_est

        int8_opt_metrics = compute_metrics_at_threshold(y_test, test_probs_int8, t_int8_opt)
        int8_default_metrics = compute_metrics_at_threshold(y_test, test_probs_int8, 0.50)

        per_seed_results.append({
            "seed": seed,
            "fp32_at_val_threshold": fp32_opt_metrics,
            "fp32_at_default_0_5": fp32_default_metrics,
            "int8_at_val_threshold": int8_opt_metrics,
            "int8_at_default_0_5": int8_default_metrics,
            "drop_at_val_threshold": {
                "accuracy_drop": round(fp32_opt_metrics["accuracy"] - int8_opt_metrics["accuracy"], 4),
                "f1_drop": round(fp32_opt_metrics["f1"] - int8_opt_metrics["f1"], 4)
            },
            "drop_at_default_0_5": {
                "accuracy_drop": round(fp32_default_metrics["accuracy"] - int8_default_metrics["accuracy"], 4),
                "f1_drop": round(fp32_default_metrics["f1"] - int8_default_metrics["f1"], 4)
            }
        })

    # Summary across seeds
    n_seeds = len(per_seed_results)
    if n_seeds == 0:
        raise RuntimeError("No seeds evaluated!")

    def mean_std(key_path, sub_key):
        vals = [r[key_path][sub_key] for r in per_seed_results]
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    # Calculate compression against TFLite FP32 (Item 4)
    compression_ratio_vs_tflite_fp32 = round(float(fp32_tflite_size / int8_tflite_size), 2)
    compression_percent_vs_tflite_fp32 = round(float((1.0 - int8_tflite_size / fp32_tflite_size) * 100.0), 2)

    summary = {
        "protocol": args.protocol,
        "model": "M1",
        "mode": args.mode,
        "n_seeds": n_seeds,
        "sizes": {
            "tflite_fp32_bytes": fp32_tflite_size,
            "tflite_fp32_kb": round(fp32_tflite_size / 1024.0, 2),
            "tflite_int8_bytes": int8_tflite_size,
            "tflite_int8_kb": round(int8_tflite_size / 1024.0, 2),
            "compression_ratio_vs_tflite_fp32": compression_ratio_vs_tflite_fp32,
            "compression_percent_vs_tflite_fp32": compression_percent_vs_tflite_fp32,
            "note_on_keras_size": "Keras .keras file contains uncompressed python metadata; TFLite FP32 is the true baseline"
        },
        "arena_estimate": {
            "estimated_arena_bytes": arena_estimate_bytes,
            "estimated_arena_kb": round(arena_estimate_bytes / 1024.0, 2),
            "note": "Estimated minimum tensor arena for TFLM runtime buffers on ESP32"
        },
        "aggregated_performance": {
            "fp32_at_val_threshold": {
                "accuracy_mean": mean_std("fp32_at_val_threshold", "accuracy")[0],
                "accuracy_std": mean_std("fp32_at_val_threshold", "accuracy")[1],
                "f1_mean": mean_std("fp32_at_val_threshold", "f1")[0],
                "f1_std": mean_std("fp32_at_val_threshold", "f1")[1]
            },
            "int8_at_val_threshold": {
                "accuracy_mean": mean_std("int8_at_val_threshold", "accuracy")[0],
                "accuracy_std": mean_std("int8_at_val_threshold", "accuracy")[1],
                "f1_mean": mean_std("int8_at_val_threshold", "f1")[0],
                "f1_std": mean_std("int8_at_val_threshold", "f1")[1]
            },
            "fp32_at_default_0_5": {
                "accuracy_mean": mean_std("fp32_at_default_0_5", "accuracy")[0],
                "accuracy_std": mean_std("fp32_at_default_0_5", "accuracy")[1],
                "f1_mean": mean_std("fp32_at_default_0_5", "f1")[0],
                "f1_std": mean_std("fp32_at_default_0_5", "f1")[1]
            },
            "int8_at_default_0_5": {
                "accuracy_mean": mean_std("int8_at_default_0_5", "accuracy")[0],
                "accuracy_std": mean_std("int8_at_default_0_5", "accuracy")[1],
                "f1_mean": mean_std("int8_at_default_0_5", "f1")[0],
                "f1_std": mean_std("int8_at_default_0_5", "f1")[1]
            },
            "drop_fp32_to_int8_mean": {
                "at_val_threshold_accuracy_drop": round(float(np.mean([r["drop_at_val_threshold"]["accuracy_drop"] for r in per_seed_results])), 4),
                "at_default_0_5_accuracy_drop": round(float(np.mean([r["drop_at_default_0_5"]["accuracy_drop"] for r in per_seed_results])), 4)
            }
        },
        "per_seed_results": per_seed_results,
        "hardware_measurements": {
            "esp32_latency_ms": "not measured here (requires physical hardware bench)",
            "esp32_power_mw": "not measured here (requires physical hardware bench)"
        }
    }

    quant_json_path = os.path.join(results_dir, "quantization.json")
    with open(quant_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n>>> Full 5-seed INT8 quantization report written to {quant_json_path}")

if __name__ == "__main__":
    main()
