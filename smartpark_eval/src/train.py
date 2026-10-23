import argparse
import json
import os
import random
import sys
import numpy as np
import tensorflow as tf
import yaml

sys.path.insert(0, os.path.abspath("."))
from smartpark_eval.src.dataset import get_or_create_cached_split, FastBatchGenerator
from smartpark_eval.src.model import build_model_m0, build_model_m1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def create_tf_dataset(images_u8, labels, batch_size=64, is_train=True, seed=42):
    gen = FastBatchGenerator(images_u8, labels, batch_size=batch_size, is_train=is_train, seed=seed)
    n_steps = len(gen)

    def generator():
        yield from gen.get_batches()

    output_signature = (
        tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, n_steps

def train_single_seed(protocol_csv, protocol_name, model_type, mode, seed, cfg):
    set_seed(seed)
    degrade = (mode == "degraded")
    batch_size = cfg["training"]["batch_size"]
    max_epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["patience"]
    lr = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]

    models_dir = os.path.join(cfg["paths"]["results_dir"], "models")
    histories_dir = os.path.join(cfg["paths"]["results_dir"], "histories")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(histories_dir, exist_ok=True)

    model_save_path = os.path.join(models_dir, f"{protocol_name}_{model_type}_{mode}_seed{seed}.keras")
    history_save_path = os.path.join(histories_dir, f"{protocol_name}_{model_type}_{mode}_seed{seed}.json")

    # 1. Load cached splits
    x_train, y_train, _ = get_or_create_cached_split(protocol_csv, "train", degrade=degrade)
    x_val, y_val, _ = get_or_create_cached_split(protocol_csv, "val", degrade=degrade)

    # 2. Class weights
    n_total = len(y_train)
    n_0 = max(1, int((y_train == 0).sum()))
    n_1 = max(1, int((y_train == 1).sum()))
    class_weights = {
        0: float(n_total / (2.0 * n_0)),
        1: float(n_total / (2.0 * n_1))
    }
    print(f"[{protocol_name} | {model_type} | {mode} | seed {seed}] Train: {n_total} (Free: {n_0}, Occ: {n_1}), Class weights: {class_weights}")

    # 3. Create tf.data datasets
    train_ds, train_steps = create_tf_dataset(x_train, y_train, batch_size=batch_size, is_train=True, seed=seed)
    val_ds, val_steps = create_tf_dataset(x_val, y_val, batch_size=batch_size, is_train=False, seed=seed)

    # 4. Build Model
    if model_type.upper() == "M0":
        model = build_model_m0()
    elif model_type.upper() == "M1":
        model = build_model_m1()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Apply weight decay
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # 5. Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # 6. Fit
    history = model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        epochs=max_epochs,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # 7. Save model and history
    model.save(model_save_path)
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_save_path, "w") as f:
        json.dump(hist_dict, f, indent=2)

    # 8. Sanity check (Section 7)
    final_train_acc = hist_dict["accuracy"][-1]
    final_val_acc = hist_dict["val_accuracy"][-1]
    print(f"Finished seed {seed}: best val_loss={min(hist_dict['val_loss']):.4f}, train_acc={final_train_acc:.4f}, val_acc={final_val_acc:.4f}")

    if final_val_acc < 0.60:
        print(f"WARNING: Validation accuracy ({final_val_acc:.4f}) is low! Inspect curves!")

    return hist_dict

def main():
    parser = argparse.ArgumentParser(description="Train SmartPark CNN models.")
    parser.add_argument("--config", default="smartpark_eval/configs/default.yaml", help="Path to config file.")
    parser.add_argument("--protocol", default="protocol_b", help="Protocol split name (e.g. protocol_b, protocol_a).")
    parser.add_argument("--model", choices=["M0", "M1"], default="M1", help="Model architecture.")
    parser.add_argument("--mode", choices=["clean", "degraded"], default="clean", help="Evaluation mode.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="List of random seeds.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    splits_dir = cfg["paths"]["splits_dir"]
    protocol_csv = os.path.join(splits_dir, f"{args.protocol}.csv")
    if not os.path.exists(protocol_csv):
        raise FileNotFoundError(f"Protocol split CSV not found: {protocol_csv}")

    for s in args.seeds:
        print(f"\n==========================================")
        print(f"Starting Training: {args.protocol} | Model: {args.model} | Mode: {args.mode} | Seed: {s}")
        print(f"==========================================")
        train_single_seed(protocol_csv, args.protocol, args.model, args.mode, s, cfg)

if __name__ == "__main__":
    main()
