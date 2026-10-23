import glob
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import yaml

sys.path.insert(0, os.path.abspath("."))

def load_config(config_path="smartpark_eval/configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_training_curves(histories_dir, figures_dir):
    """
    Plots training curves (train vs val loss & accuracy) for all history files.
    """
    os.makedirs(figures_dir, exist_ok=True)
    history_files = glob.glob(os.path.join(histories_dir, "*.json"))
    if not history_files:
        print("No history files found to plot.")
        return

    # Group histories by experiment prefix (ignore seed)
    grouped = {}
    for hf in history_files:
        base = os.path.basename(hf).replace(".json", "")
        # e.g. protocol_b_M1_clean_seed0
        parts = base.split("_seed")
        prefix = parts[0]
        if prefix not in grouped:
            grouped[prefix] = []
        with open(hf, "r") as f:
            grouped[prefix].append(json.load(f))

    for prefix, hist_list in grouped.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
        
        # Plot Loss
        ax_loss = axes[0]
        for hist in hist_list:
            epochs = range(1, len(hist["loss"]) + 1)
            ax_loss.plot(epochs, hist["loss"], color="tab:blue", alpha=0.3, label="Train" if "Train" not in [l.get_label() for l in ax_loss.lines] else "")
            ax_loss.plot(epochs, hist["val_loss"], color="tab:orange", alpha=0.3, label="Val" if "Val" not in [l.get_label() for l in ax_loss.lines] else "")
        ax_loss.set_title(f"{prefix} - Loss", fontsize=10)
        ax_loss.set_xlabel("Epoch", fontsize=9)
        ax_loss.set_ylabel("Binary Cross-Entropy", fontsize=9)
        ax_loss.grid(True, linestyle="--", alpha=0.5)
        ax_loss.legend(fontsize=8)

        # Plot Accuracy
        ax_acc = axes[1]
        for hist in hist_list:
            epochs = range(1, len(hist["accuracy"]) + 1)
            ax_acc.plot(epochs, hist["accuracy"], color="tab:blue", alpha=0.3, label="Train" if "Train" not in [l.get_label() for l in ax_acc.lines] else "")
            ax_acc.plot(epochs, hist["val_accuracy"], color="tab:orange", alpha=0.3, label="Val" if "Val" not in [l.get_label() for l in ax_acc.lines] else "")
        ax_acc.set_title(f"{prefix} - Accuracy", fontsize=10)
        ax_acc.set_xlabel("Epoch", fontsize=9)
        ax_acc.set_ylabel("Accuracy", fontsize=9)
        ax_acc.grid(True, linestyle="--", alpha=0.5)
        ax_acc.legend(fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(figures_dir, f"training_curve_{prefix}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved training curve figure: {out_path}")

def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plots and saves normalized confusion matrix."""
    cm = np.array(cm)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=300)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix: {save_path}")

def generate_latex_tables(results_dir, latex_dir):
    """
    Generates booktabs LaTeX tables ready for IEEE conference paper.
    """
    os.makedirs(latex_dir, exist_ok=True)

    # 1. Splits table
    split_summary_path = os.path.join(results_dir, "split_summary.csv")
    if os.path.exists(split_summary_path):
        df_splits = pd.read_csv(split_summary_path)
        tex_splits = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Summary of Datasets and Leak-Free Evaluation Partitions}",
            "\\label{tab:datasets_splits}",
            "\\begin{tabular}{llrrrr}",
            "\\toprule",
            "\\textbf{Protocol} & \\textbf{Split} & \\textbf{Samples} & \\textbf{Free} & \\textbf{Occupied} & \\textbf{Occ. Ratio} \\\\",
            "\\midrule"
        ]
        for _, row in df_splits.iterrows():
            tex_splits.append(
                f"{row['protocol']} & {row['split']} & {int(row['total_samples']):,} & "
                f"{int(row['free_count']):,} & {int(row['occupied_count']):,} & {row['occupied_ratio']:.3f} \\\\"
            )
        tex_splits.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        with open(os.path.join(latex_dir, "table_datasets_splits.tex"), "w") as f:
            f.write("\n".join(tex_splits))
        print("Generated table_datasets_splits.tex")

    # 2. Main results table
    metrics_path = os.path.join(results_dir, "metrics_all.csv")
    if os.path.exists(metrics_path):
        df_m = pd.read_csv(metrics_path)
        tex_main = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{SmartPark Occupancy Classification Benchmark Across Protocols (Mean $\\pm$ Std over 5 Seeds)}",
            "\\label{tab:main_results}",
            "\\begin{tabular}{lllcccccc}",
            "\\toprule",
            "\\textbf{Protocol} & \\textbf{Model} & \\textbf{Mode} & \\textbf{Accuracy} & \\textbf{Balanced Acc.} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{95\\% CI} \\\\",
            "\\midrule"
        ]
        for _, row in df_m.iterrows():
            acc_str = f"{row['acc_mean']:.3f} $\\pm$ {row['acc_std']:.3f}"
            bacc_str = f"{row['bacc_mean']:.3f} $\\pm$ {row['bacc_std']:.3f}"
            prec_str = f"{row['prec_mean']:.3f} $\\pm$ {row['prec_std']:.3f}"
            rec_str = f"{row['rec_mean']:.3f} $\\pm$ {row['rec_std']:.3f}"
            f1_str = f"{row['f1_mean']:.3f} $\\pm$ {row['f1_std']:.3f}"
            ci_str = f"[{row['ci_lo_mean']:.3f}, {row['ci_hi_mean']:.3f}]"
            tex_main.append(
                f"{row['protocol']} & {row['model']} & {row['mode']} & {acc_str} & {bacc_str} & {prec_str} & {rec_str} & {f1_str} & {ci_str} \\\\"
            )
        tex_main.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}"
        ])
        with open(os.path.join(latex_dir, "table_main_results.tex"), "w") as f:
            f.write("\n".join(tex_main))
        print("Generated table_main_results.tex")

    # 3. Ablation table
    ablation_path = os.path.join(results_dir, "ablation.csv")
    if os.path.exists(ablation_path):
        df_abl = pd.read_csv(ablation_path)
        tex_abl = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Ablation Study: Algorithmic Fallback vs. Neural Architectures}",
            "\\label{tab:ablation}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "\\textbf{Method} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} \\\\",
            "\\midrule"
        ]
        for _, row in df_abl.iterrows():
            acc = f"{row['accuracy']}" if isinstance(row['accuracy'], str) else f"{float(row['accuracy']):.3f}"
            prec = f"{row['precision']}" if isinstance(row['precision'], str) else f"{float(row['precision']):.3f}"
            rec = f"{row['recall']}" if isinstance(row['recall'], str) else f"{float(row['recall']):.3f}"
            f1 = f"{row['f1']}" if isinstance(row['f1'], str) else f"{float(row['f1']):.3f}"
            tex_abl.append(f"{row['method']} & {acc} & {prec} & {rec} & {f1} \\\\")
        tex_abl.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        with open(os.path.join(latex_dir, "table_ablation.tex"), "w") as f:
            f.write("\n".join(tex_abl))
        print("Generated table_ablation.tex")

    # 4. Quantization table
    quant_path = os.path.join(results_dir, "quantization.json")
    if os.path.exists(quant_path):
        with open(quant_path, "r") as f:
            q = json.load(f)
        s = q["sizes"]
        p = q["performance_comparison"]
        tex_q = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Post-Training Quantization Footprint and Performance}",
            "\\label{tab:quantization}",
            "\\begin{tabular}{lrr}",
            "\\toprule",
            "\\textbf{Metric / Artifact} & \\textbf{FP32 Model} & \\textbf{INT8 Model} \\\\",
            "\\midrule",
            f"Keras / Source File Size & {s['keras_fp32_kb']:.1f} KB & --- \\\\",
            f"TFLite Model File Size & {s['tflite_fp32_kb']:.1f} KB & \\textbf{{{s['tflite_int8_kb']:.1f} KB}} \\\\",
            f"Compression Ratio & 1.0$\\times$ & \\textbf{{{s['compression_ratio_vs_keras']:.1f}$\\times$}} ({s['compression_percent']:.1f}\\% reduction) \\\\",
            f"Estimated Tensor Arena & --- & {q['arena_estimate']['estimated_arena_kb']:.1f} KB \\\\",
            f"Test Accuracy & {p['fp32_accuracy']:.3f} & {p['int8_accuracy']:.3f} ($\\Delta$ = {p['accuracy_drop_fp32_to_int8']:.3f}) \\\\",
            f"Test F1-Score & {p['fp32_f1']:.3f} & {p['int8_f1']:.3f} ($\\Delta$ = {p['f1_drop_fp32_to_int8']:.3f}) \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ]
        with open(os.path.join(latex_dir, "table_quantization.tex"), "w") as f:
            f.write("\n".join(tex_q))
        print("Generated table_quantization.tex")

def main():
    cfg = load_config("smartpark_eval/configs/default.yaml")
    results_dir = cfg["paths"]["results_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    latex_dir = cfg["paths"]["latex_dir"]
    histories_dir = os.path.join(results_dir, "histories")

    print("Generating training curves...")
    generate_training_curves(histories_dir, figures_dir)

    print("Generating LaTeX tables...")
    generate_latex_tables(results_dir, latex_dir)

    print("Report generation complete!")

if __name__ == "__main__":
    main()
