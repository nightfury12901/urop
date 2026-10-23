import os
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model_m0(input_shape=(64, 64, 3)):
    """
    M0 - Paper baseline architecture for honest comparison:
    Conv(16)-Pool, Conv(32)-Pool, Conv(64)-Pool, all 3x3 ReLU;
    Flatten (8x8x64 = 4096); Dense(64) ReLU; Dropout(0.5); Dense(1, sigmoid).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="input_1"),
        layers.Conv2D(16, 3, padding="same", activation="relu", name="conv2d_1"),
        layers.MaxPooling2D((2, 2), name="max_pooling2d_1"),
        layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2d_2"),
        layers.MaxPooling2D((2, 2), name="max_pooling2d_2"),
        layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2d_3"),
        layers.MaxPooling2D((2, 2), name="max_pooling2d_3"),
        layers.Flatten(name="flatten_1"),
        layers.Dense(64, activation="relu", name="dense_1"),
        layers.Dropout(0.5, name="dropout_1"),
        layers.Dense(1, activation="sigmoid", name="output_dense")
    ], name="SmartPark_M0_Baseline")
    return model

def build_model_m1(input_shape=(64, 64, 3), filters=(16, 32, 64)):
    """
    M1 - Compact model designed for ESP32 deployment:
    Conv-BN-Pool x3, GlobalAveragePooling2D instead of Flatten+Dense,
    Dropout(0.3), Dense(1, sigmoid).
    With filters=(16, 32, 64): total params = 24,097 (under 25k).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="input_1"),
        layers.Conv2D(filters[0], 3, padding="same", activation="relu", name="conv2d_1"),
        layers.BatchNormalization(name="batch_norm_1"),
        layers.MaxPooling2D((2, 2), name="max_pooling2d_1"),
        layers.Conv2D(filters[1], 3, padding="same", activation="relu", name="conv2d_2"),
        layers.BatchNormalization(name="batch_norm_2"),
        layers.MaxPooling2D((2, 2), name="max_pooling2d_2"),
        layers.Conv2D(filters[2], 3, padding="same", activation="relu", name="conv2d_3"),
        layers.BatchNormalization(name="batch_norm_3"),
        layers.MaxPooling2D((2, 2), name="max_pooling2d_3"),
        layers.GlobalAveragePooling2D(name="global_avg_pool_1"),
        layers.Dropout(0.3, name="dropout_1"),
        layers.Dense(1, activation="sigmoid", name="output_dense")
    ], name=f"SmartPark_M1_Compact_{filters[0]}_{filters[1]}_{filters[2]}")
    return model

def generate_model_markdown(model, model_name, output_path):
    """
    Generates a GitHub-flavored Markdown layer table and parameter summary.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_params = model.count_params()
    trainable_params = sum(p.numpy().size for p in model.trainable_variables)
    non_trainable_params = total_params - trainable_params
    fp32_size_bytes = total_params * 4
    fp32_size_kb = fp32_size_bytes / 1024.0

    lines = [
        f"# Model Architecture: {model_name}",
        f"",
        f"- **Model Class:** `{model.name}`",
        f"- **Total Parameters:** {total_params:,}",
        f"- **Trainable Parameters:** {trainable_params:,}",
        f"- **Non-trainable Parameters:** {non_trainable_params:,}",
        f"- **Calculated Weights FP32 Size:** {fp32_size_bytes:,} bytes ({fp32_size_kb:.2f} KB)",
        f"",
        "| Layer Index | Layer Name | Layer Type | Output Shape | Param # |",
        "| :--- | :--- | :--- | :--- | ---: |"
    ]

    for idx, layer in enumerate(model.layers):
        output_shape = str(layer.output.shape) if hasattr(layer, "output") else "N/A"
        param_count = layer.count_params()
        layer_type = layer.__class__.__name__
        lines.append(f"| {idx + 1} | `{layer.name}` | {layer_type} | `{output_shape}` | {param_count:,} |")

    lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Generated layer table at {output_path}")

if __name__ == "__main__":
    results_dir = "smartpark_eval/results"
    os.makedirs(results_dir, exist_ok=True)

    m0 = build_model_m0()
    m0.summary()
    generate_model_markdown(m0, "M0 (Paper Baseline)", os.path.join(results_dir, "model_M0.md"))

    m1 = build_model_m1()
    m1.summary()
    generate_model_markdown(m1, "M1 (Compact Target)", os.path.join(results_dir, "model_M1.md"))
