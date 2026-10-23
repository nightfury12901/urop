"""
generate_final_figures_package.py
=================================
Generates all publication-ready figures at 300 DPI in a dedicated folder:
  `smartpark_eval/final_figures/`

Includes:
  1. Fig_1_Training_Curves_M1_clean (PDF & PNG)
  2. Fig_2_Training_Curves_M0_clean (PDF & PNG)
  3. Fig_3_Training_Curves_M1_degraded (PDF & PNG)
  4. Fig_4_Confusion_Matrix_M1_clean_best_seed3 (PDF & PNG)
  5. Fig_5_Confusion_Matrix_M1_clean_seed1 (PDF & PNG)
  6. Fig_6_Confusion_Matrix_M0_baseline_seed0 (PDF & PNG)
  7. Fig_7_Confusion_Matrix_M1_degraded_seed4 (PDF & PNG)
  8. Fig_8_Worst_False_Positives_and_Negatives (PDF & PNG)
  9. Fig_9_Quantization_Compression_Tradeoff (PDF & PNG)
"""

import os, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as skm_cm

sys.path.insert(0, os.path.abspath("."))
from smartpark_eval.src.dataset import get_or_create_cached_split

OUT_DIR = "smartpark_eval/final_figures"
HISTORIES_DIR = "smartpark_eval/results/histories"
MODELS_DIR = "smartpark_eval/results/models"
SPLITS_CSV = "smartpark_eval/splits/protocol_b.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# Plot styling for IEEE papers
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})

# 1. Training Curves (Aggregated across 5 seeds with shaded std)
def plot_training_curves(prefix, title, out_name):
    files = [f for f in os.listdir(HISTORIES_DIR) if f.startswith(prefix) and f.endswith(".json")]
    if not files:
        print(f"Skipping {prefix}: no history files found.")
        return

    all_losses, all_val_losses = [], []
    all_accs, all_val_accs = [], []
    max_epochs = 0

    for fname in sorted(files):
        with open(os.path.join(HISTORIES_DIR, fname), encoding="utf-8") as f:
            h = json.load(f)
        all_losses.append(h["loss"])
        all_val_losses.append(h["val_loss"])
        all_accs.append(h["accuracy"])
        all_val_accs.append(h["val_accuracy"])
        max_epochs = max(max_epochs, len(h["loss"]))

    # Pad arrays to same length with nan for averaging
    def pad(arr_list):
        arr = np.full((len(arr_list), max_epochs), np.nan)
        for i, a in enumerate(arr_list):
            arr[i, :len(a)] = a
        return arr

    losses = pad(all_losses)
    val_losses = pad(all_val_losses)
    accs = pad(all_accs)
    val_accs = pad(all_val_accs)

    epochs = np.arange(1, max_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=300)

    # Loss plot
    mean_l, std_l = np.nanmean(losses, axis=0), np.nanstd(losses, axis=0)
    mean_vl, std_vl = np.nanmean(val_losses, axis=0), np.nanstd(val_losses, axis=0)

    ax1.plot(epochs, mean_l, color="#1f77b4", lw=2, label="Train Loss (Mean)")
    ax1.fill_between(epochs, mean_l - std_l, mean_l + std_l, color="#1f77b4", alpha=0.18)
    ax1.plot(epochs, mean_vl, color="#ff7f0e", lw=2, linestyle="--", label="Val Loss (Mean)")
    ax1.fill_between(epochs, mean_vl - std_vl, mean_vl + std_vl, color="#ff7f0e", alpha=0.18)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Accuracy plot
    mean_a, std_a = np.nanmean(accs, axis=0), np.nanstd(accs, axis=0)
    mean_va, std_va = np.nanmean(val_accs, axis=0), np.nanstd(val_accs, axis=0)

    ax2.plot(epochs, mean_a, color="#1f77b4", lw=2, label="Train Accuracy (Mean)")
    ax2.fill_between(epochs, mean_a - std_a, mean_a + std_a, color="#1f77b4", alpha=0.18)
    ax2.plot(epochs, mean_va, color="#ff7f0e", lw=2, linestyle="--", label="Val Accuracy (Mean)")
    ax2.fill_between(epochs, mean_va - std_va, mean_va + std_va, color="#ff7f0e", alpha=0.18)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} - Accuracy")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right")
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    pdf_path = os.path.join(OUT_DIR, f"{out_name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{out_name}.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path} & {pdf_path}")

# 2. Confusion Matrix Plot
def plot_confusion_matrix(cm, title, out_name, acc=None, f1=None):
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=300)
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = ["Free", "Occupied"]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            txt_color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=txt_color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Ground Truth Label", fontsize=11)

    title_str = title
    if acc is not None:
        title_str += f"\nAcc: {acc*100:.2f}% | F1: {f1*100:.2f}%"
    ax.set_title(title_str, fontsize=11)

    plt.tight_layout()
    pdf_path = os.path.join(OUT_DIR, f"{out_name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{out_name}.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")

# 3. Quantization Tradeoff Bar Chart
def plot_quantization_tradeoff():
    with open("smartpark_eval/results/quantization.json", encoding="utf-8") as f:
        q = json.load(f)

    models = ["Baseline M0\n(285k params)", "Proposed M1 (FP32)\n(24k params)", "Proposed M1 (INT8)\n(TFLite 8-bit)"]
    sizes_kb = [1093.4, q["sizes"]["tflite_fp32_kb"], q["sizes"]["tflite_int8_kb"]]
    accs = [96.38, q["aggregated_performance"]["fp32_at_default_0_5"]["accuracy_mean"] * 100,
            q["aggregated_performance"]["int8_at_default_0_5"]["accuracy_mean"] * 100]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2), dpi=300)

    x = np.arange(len(models))
    width = 0.35

    color1 = "#2ca02c"
    color2 = "#1f77b4"

    # Bar for size
    rects1 = ax1.bar(x - width/2, sizes_kb, width, label="Model Size (KB, log scale)", color=color1, alpha=0.85)
    ax1.set_ylabel("Model Size on Disk (KB)", color=color1, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)

    # Bar for accuracy
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, accs, width, label="Accuracy (%)", color=color2, alpha=0.85)
    ax2.set_ylabel("Classification Accuracy (%)", color=color2, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([85, 100])

    # Annotations
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f} KB", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for r in rects2:
        h = r.get_height()
        ax2.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.title("SmartPark Model Size Reduction vs. Classification Accuracy\n(33.5x size reduction from M0 with < 1.4% accuracy delta)", fontsize=11)
    plt.tight_layout()

    pdf_path = os.path.join(OUT_DIR, "Fig_Quantization_Tradeoff.pdf")
    png_path = os.path.join(OUT_DIR, "Fig_Quantization_Tradeoff.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")

# 4. Error Grids (Worst False Positives and Negatives)
def generate_error_grids():
    import tensorflow as tf
    mpath = "smartpark_eval/results/models/protocol_b_M1_clean_seed0.keras"
    if not os.path.exists(mpath):
        return

    x_test_u8, y_test, _ = get_or_create_cached_split(SPLITS_CSV, "test", degrade=False)
    model = tf.keras.models.load_model(mpath)
    probs = model.predict(x_test_u8.astype(np.float32) / 255.0, batch_size=256, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)

    fp_idx = np.where((y_test == 0) & (preds == 1))[0]
    fp_idx = fp_idx[np.argsort(-probs[fp_idx])][:8]

    fn_idx = np.where((y_test == 1) & (preds == 0))[0]
    fn_idx = fn_idx[np.argsort(probs[fn_idx])][:8]

    fig, axes = plt.subplots(2, 8, figsize=(14, 4.2), dpi=300)

    for i in range(8):
        ax = axes[0, i]
        if i < len(fp_idx):
            idx = fp_idx[i]
            ax.imshow(x_test_u8[idx])
            ax.set_title(f"FP (p={probs[idx]:.2f})", fontsize=8)
        ax.axis("off")
        if i == 0:
            ax.text(-0.3, 0.5, "False Positives\n(Empty -> Occupied)", transform=ax.transAxes,
                    va="center", ha="right", fontsize=9, fontweight="bold")

    for i in range(8):
        ax = axes[1, i]
        if i < len(fn_idx):
            idx = fn_idx[i]
            ax.imshow(x_test_u8[idx])
            ax.set_title(f"FN (p={probs[idx]:.2f})", fontsize=8)
        ax.axis("off")
        if i == 0:
            ax.text(-0.3, 0.5, "False Negatives\n(Occupied -> Empty)", transform=ax.transAxes,
                    va="center", ha="right", fontsize=9, fontweight="bold")

    plt.suptitle("Top Failure Modes on Camera 9 Test Set (Challenging Shadows & Occlusions)", fontsize=11)
    plt.tight_layout()
    pdf_path = os.path.join(OUT_DIR, "Fig_Worst_Errors_Grid.pdf")
    png_path = os.path.join(OUT_DIR, "Fig_Worst_Errors_Grid.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")

def main():
    print("Generating publication figures into smartpark_eval/final_figures/...")

    # 1. Training curves (5 seeds mean + std band)
    plot_training_curves("protocol_b_M1_clean", "Proposed M1 (Clean)", "Fig_Training_Curves_M1_Clean")
    plot_training_curves("protocol_b_M0_clean", "Baseline M0 (Clean)", "Fig_Training_Curves_M0_Clean")
    plot_training_curves("protocol_b_M1_degraded", "Proposed M1 (Degraded)", "Fig_Training_Curves_M1_Degraded")

    # 2. Confusion Matrices
    # Best M1 clean seed 3 (97.03% acc)
    cm_seed3 = np.array([[3876, 192], [85, 5182]])
    plot_confusion_matrix(cm_seed3, "M1 Proposed (Clean) - Seed 3 [Best]\nHeld-Out Camera 9 (9,335 patches)",
                          "Fig_Confusion_Matrix_M1_Clean_Best_Seed3", acc=0.9703, f1=0.9741)

    # Median M1 clean seed 1 (95.99% acc)
    cm_seed1 = np.array([[3778, 290], [84, 5183]])
    plot_confusion_matrix(cm_seed1, "M1 Proposed (Clean) - Seed 1 [Representative]\nHeld-Out Camera 9 (9,335 patches)",
                          "Fig_Confusion_Matrix_M1_Clean_Seed1", acc=0.9599, f1=0.9648)

    # M0 baseline seed 0 (96.80% acc)
    cm_m0 = np.array([[3894, 174], [125, 5142]])
    plot_confusion_matrix(cm_m0, "M0 Baseline (Clean) - Seed 0\nHeld-Out Camera 9 (9,335 patches)",
                          "Fig_Confusion_Matrix_M0_Baseline_Seed0", acc=0.9680, f1=0.9715)

    # M1 degraded seed 3 (95.27% acc)
    cm_deg = np.array([[3742, 326], [116, 5151]])
    plot_confusion_matrix(cm_deg, "M1 Proposed (Degraded) - Seed 3\nHeld-Out Camera 9 (9,335 patches)",
                          "Fig_Confusion_Matrix_M1_Degraded_Seed3", acc=0.9527, f1=0.9590)

    # 3. Quantization Tradeoff chart
    plot_quantization_tradeoff()

    # 4. Error grids
    generate_error_grids()

    print("\nAll publication figures successfully created in smartpark_eval/final_figures/!")

if __name__ == "__main__":
    main()
