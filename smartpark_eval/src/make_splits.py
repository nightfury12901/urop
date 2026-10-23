import argparse
import os
import numpy as np
import pandas as pd
import yaml

def load_config(config_path="smartpark_eval/configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def assert_leak_free(df_split, protocol_name):
    """
    Asserts strict, leak-free partition integrity:
    1. No file path appears twice across splits.
    2. No (camera, date) pair appears in more than one split (unless explicitly test-held-out camera with separate dates).
    3. No cross-dataset contamination.
    """
    splits = df_split["split"].unique()
    paths_by_split = {s: set(df_split[df_split["split"] == s]["path"]) for s in splits}
    
    # Path disjointness
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = paths_by_split[s1] & paths_by_split[s2]
            assert len(overlap) == 0, f"[{protocol_name}] Leakage detected: {len(overlap)} paths shared between {s1} and {s2}!"

    # Group disjointness: (camera, date) disjointness
    # Train and Val must NEVER share dates
    train_groups = set(zip(df_split[df_split["split"] == "train"]["camera"], df_split[df_split["split"] == "train"]["date"]))
    val_groups = set(zip(df_split[df_split["split"] == "val"]["camera"], df_split[df_split["split"] == "val"]["date"]))
    train_val_overlap = train_groups & val_groups
    assert len(train_val_overlap) == 0, f"[{protocol_name}] Leakage: (camera, date) shared between train and val: {train_val_overlap}!"

    # Test set checks
    for test_split_name in [s for s in splits if "test" in s]:
        test_groups = set(zip(df_split[df_split["split"] == test_split_name]["camera"], df_split[df_split["split"] == test_split_name]["date"]))
        train_test_overlap = train_groups & test_groups
        val_test_overlap = val_groups & test_groups
        assert len(train_test_overlap) == 0, f"[{protocol_name}] Leakage: (camera, date) shared between train and {test_split_name}: {train_test_overlap}!"
        assert len(val_test_overlap) == 0, f"[{protocol_name}] Leakage: (camera, date) shared between val and {test_split_name}: {val_test_overlap}!"

    # Cross-dataset check if applicable
    if "c1" in protocol_name.lower():
        train_datasets = set(df_split[df_split["split"] == "train"]["dataset"])
        test_datasets = set(df_split[df_split["split"].str.startswith("test")]["dataset"])
        assert train_datasets.isdisjoint(test_datasets), f"[{protocol_name}] Cross-dataset contamination!"
    elif "c2" in protocol_name.lower():
        train_datasets = set(df_split[df_split["split"] == "train"]["dataset"])
        test_datasets = set(df_split[df_split["split"].str.startswith("test")]["dataset"])
        assert train_datasets.isdisjoint(test_datasets), f"[{protocol_name}] Cross-dataset contamination!"

    print(f"Leakage assertions passed for {protocol_name}.")

def make_protocol_b(df_cnr, splits_dir):
    """
    Protocol B - CNRPark-EXT, held-out camera:
    - Train: cameras 1-8, with validation as whole dates held out (20% of dates).
    - Test: camera 9 (all dates).
    """
    # Cameras 1-8
    c1_8 = df_cnr[df_cnr["camera"].isin([f"camera{i}" for i in range(1, 9)])].copy()
    c9 = df_cnr[df_cnr["camera"] == "camera9"].copy()

    # Deterministic split of dates (20% val dates, sorted with fixed seed)
    unique_dates = sorted(c1_8["date"].unique())
    rng = np.random.default_rng(42)
    shuffled_dates = list(unique_dates)
    rng.shuffle(shuffled_dates)

    n_val_dates = max(1, int(round(len(shuffled_dates) * 0.20)))
    val_dates = set(shuffled_dates[:n_val_dates])
    train_dates = set(shuffled_dates[n_val_dates:])

    c1_8["split"] = c1_8["date"].apply(lambda d: "val" if d in val_dates else "train")
    c9["split"] = "test"

    df_b = pd.concat([c1_8, c9], ignore_index=True)
    cols = ["path", "label", "camera", "date", "weather", "hour", "dataset", "split"]
    df_b = df_b[cols]

    assert_leak_free(df_b, "Protocol B")
    out_path = os.path.join(splits_dir, "protocol_b.csv")
    df_b.to_csv(out_path, index=False)
    print(f"Saved Protocol B split to {out_path} ({len(df_b)} samples).")
    return df_b

def make_protocol_b_author(cnr_dir, df_cnr_all, splits_dir):
    """
    Protocol B secondary - Author's split from splits.zip.
    """
    author_splits_dir = os.path.join(cnr_dir, "splits", "splits", "CNRPark-EXT")
    if not os.path.exists(author_splits_dir):
        print("Author splits not found, skipping author split generation.")
        return None

    path_to_split = {}
    for split_name in ["train", "val", "test"]:
        split_file = os.path.join(author_splits_dir, f"{split_name}.txt")
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                for line in f:
                    p = line.strip().split()[0]
                    path_to_split[p] = split_name

    df_author = df_cnr_all.copy()
    df_author["split"] = df_author["rel_path"].map(path_to_split)
    df_author = df_author.dropna(subset=["split"])
    cols = ["path", "label", "camera", "date", "weather", "hour", "dataset", "split"]
    df_author = df_author[cols]

    out_path = os.path.join(splits_dir, "protocol_b_author.csv")
    df_author.to_csv(out_path, index=False)
    print(f"Saved Protocol B Author split to {out_path} ({len(df_author)} samples).")
    return df_author

def make_protocol_a(df_pklot, splits_dir):
    """
    Protocol A - PKLot, held-out camera (primary):
    - Train: UFPR04 + UFPR05 (minus validation dates).
    - Validation: 20% of whole dates of training cameras.
    - Test: PUC (PUCPR).
    """
    train_cams = df_pklot[df_pklot["camera"].isin(["UFPR04", "UFPR05"])].copy()
    test_cam = df_pklot[df_pklot["camera"] == "PUC"].copy()

    unique_dates = sorted(train_cams["date"].unique())
    rng = np.random.default_rng(42)
    shuffled_dates = list(unique_dates)
    rng.shuffle(shuffled_dates)

    n_val_dates = max(1, int(round(len(shuffled_dates) * 0.20)))
    val_dates = set(shuffled_dates[:n_val_dates])

    train_cams["split"] = train_cams["date"].apply(lambda d: "val" if d in val_dates else "train")
    test_cam["split"] = "test"

    df_a = pd.concat([train_cams, test_cam], ignore_index=True)
    cols = ["path", "label", "camera", "date", "weather", "hour", "dataset", "split"]
    df_a = df_a[cols]

    assert_leak_free(df_a, "Protocol A")
    out_path = os.path.join(splits_dir, "protocol_a.csv")
    df_a.to_csv(out_path, index=False)
    print(f"Saved Protocol A split to {out_path} ({len(df_a)} samples).")
    return df_a

def make_protocol_c1(df_pklot, df_cnr, splits_dir):
    """
    Protocol C1 - Cross-dataset:
    - Train: PKLot (UFPR04 + UFPR05 + PUC minus 20% validation dates).
    - Validation: PKLot 20% dates.
    - Test: CNRPark-EXT (all cameras).
    """
    unique_dates = sorted(df_pklot["date"].unique())
    rng = np.random.default_rng(42)
    shuffled_dates = list(unique_dates)
    rng.shuffle(shuffled_dates)

    n_val_dates = max(1, int(round(len(shuffled_dates) * 0.20)))
    val_dates = set(shuffled_dates[:n_val_dates])

    df_pklot_copy = df_pklot.copy()
    df_pklot_copy["split"] = df_pklot_copy["date"].apply(lambda d: "val" if d in val_dates else "train")

    df_cnr_copy = df_cnr.copy()
    df_cnr_copy["split"] = "test"

    df_c1 = pd.concat([df_pklot_copy, df_cnr_copy], ignore_index=True)
    cols = ["path", "label", "camera", "date", "weather", "hour", "dataset", "split"]
    df_c1 = df_c1[cols]

    assert_leak_free(df_c1, "Protocol C1")
    out_path = os.path.join(splits_dir, "protocol_c1.csv")
    df_c1.to_csv(out_path, index=False)
    print(f"Saved Protocol C1 split to {out_path} ({len(df_c1)} samples).")
    return df_c1

def make_protocol_c2(df_cnr, df_pklot, splits_dir):
    """
    Protocol C2 - Cross-dataset:
    - Train: CNRPark-EXT (cameras 1-9 minus 20% validation dates).
    - Validation: CNRPark-EXT 20% dates.
    - Test: PKLot (all cameras).
    """
    unique_dates = sorted(df_cnr["date"].unique())
    rng = np.random.default_rng(42)
    shuffled_dates = list(unique_dates)
    rng.shuffle(shuffled_dates)

    n_val_dates = max(1, int(round(len(shuffled_dates) * 0.20)))
    val_dates = set(shuffled_dates[:n_val_dates])

    df_cnr_copy = df_cnr.copy()
    df_cnr_copy["split"] = df_cnr_copy["date"].apply(lambda d: "val" if d in val_dates else "train")

    df_pklot_copy = df_pklot.copy()
    df_pklot_copy["split"] = "test"

    df_c2 = pd.concat([df_cnr_copy, df_pklot_copy], ignore_index=True)
    cols = ["path", "label", "camera", "date", "weather", "hour", "dataset", "split"]
    df_c2 = df_c2[cols]

    assert_leak_free(df_c2, "Protocol C2")
    out_path = os.path.join(splits_dir, "protocol_c2.csv")
    df_c2.to_csv(out_path, index=False)
    print(f"Saved Protocol C2 split to {out_path} ({len(df_c2)} samples).")
    return df_c2

def make_protocol_d(df_pklot, df_cnr, splits_dir):
    """
    Protocol D - Combined training:
    - Train: PKLot train cameras (UFPR04+05 minus val dates) + CNRPark cameras 1-8 (minus val dates).
    - Validation: held-out dates of both datasets.
    - Test: PUCPR (test_puc) and camera 9 (test_cam9) separately.
    """
    # PKLot
    pklot_train_cams = df_pklot[df_pklot["camera"].isin(["UFPR04", "UFPR05"])].copy()
    pklot_test = df_pklot[df_pklot["camera"] == "PUC"].copy()
    
    unique_pklot_dates = sorted(pklot_train_cams["date"].unique())
    rng = np.random.default_rng(42)
    shuffled_pklot_dates = list(unique_pklot_dates)
    rng.shuffle(shuffled_pklot_dates)
    n_val_pklot = max(1, int(round(len(shuffled_pklot_dates) * 0.20)))
    val_pklot_dates = set(shuffled_pklot_dates[:n_val_pklot])
    pklot_train_cams["split"] = pklot_train_cams["date"].apply(lambda d: "val" if d in val_pklot_dates else "train")
    pklot_test["split"] = "test_puc"

    # CNRPark-EXT
    cnr_c1_8 = df_cnr[df_cnr["camera"].isin([f"camera{i}" for i in range(1, 9)])].copy()
    cnr_c9 = df_cnr[df_cnr["camera"] == "camera9"].copy()

    unique_cnr_dates = sorted(cnr_c1_8["date"].unique())
    shuffled_cnr_dates = list(unique_cnr_dates)
    rng.shuffle(shuffled_cnr_dates)
    n_val_cnr = max(1, int(round(len(shuffled_cnr_dates) * 0.20)))
    val_cnr_dates = set(shuffled_cnr_dates[:n_val_cnr])
    cnr_c1_8["split"] = cnr_c1_8["date"].apply(lambda d: "val" if d in val_cnr_dates else "train")
    cnr_c9["split"] = "test_cam9"

    df_d = pd.concat([pklot_train_cams, cnr_c1_8, pklot_test, cnr_c9], ignore_index=True)
    cols = ["path", "label", "camera", "date", "weather", "hour", "dataset", "split"]
    df_d = df_d[cols]

    assert_leak_free(df_d, "Protocol D")
    out_path = os.path.join(splits_dir, "protocol_d.csv")
    df_d.to_csv(out_path, index=False)
    print(f"Saved Protocol D split to {out_path} ({len(df_d)} samples).")
    return df_d

def generate_split_summary(protocol_dfs, results_dir):
    """
    Generates split_summary.csv detailing counts per split, protocol, class, and camera.
    """
    rows = []
    for proto_name, df in protocol_dfs.items():
        if df is None:
            continue
        for split_val in sorted(df["split"].unique()):
            sub = df[df["split"] == split_val]
            total_samples = len(sub)
            n_free = int((sub["label"] == 0).sum())
            n_occ = int((sub["label"] == 1).sum())
            cameras = sorted(sub["camera"].unique())
            dates = sub["date"].nunique()
            datasets = sorted(sub["dataset"].unique())
            rows.append({
                "protocol": proto_name,
                "split": split_val,
                "total_samples": total_samples,
                "free_count": n_free,
                "occupied_count": n_occ,
                "occupied_ratio": round(n_occ / total_samples, 4) if total_samples > 0 else 0,
                "num_cameras": len(cameras),
                "cameras": ";".join(cameras),
                "num_dates": dates,
                "datasets": ";".join(datasets)
            })

    summary_df = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "split_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"Saved split summary to {out_path}")
    print(summary_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Create deterministic leak-free splits.")
    parser.add_argument("--config", default="smartpark_eval/configs/default.yaml", help="Path to config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits_dir = cfg["paths"]["splits_dir"]
    results_dir = cfg["paths"]["results_dir"]
    cnr_dir = cfg["datasets"]["cnr_ext"]["extract_dir"]

    # Load sampled metadata tables
    df_cnr_sampled = pd.read_csv(os.path.join(splits_dir, "sampled_cnr_ext.csv"))
    df_cnr_all = pd.read_csv(os.path.join(splits_dir, "metadata_cnr_ext.csv"))
    df_pklot_sampled = pd.read_csv(os.path.join(splits_dir, "sampled_pklot.csv"))

    protocol_dfs = {}
    print("--- Generating Protocol B ---")
    protocol_dfs["Protocol B"] = make_protocol_b(df_cnr_sampled, splits_dir)

    print("--- Generating Protocol B (Author) ---")
    protocol_dfs["Protocol B (Author)"] = make_protocol_b_author(cnr_dir, df_cnr_all, splits_dir)

    print("--- Generating Protocol A ---")
    protocol_dfs["Protocol A"] = make_protocol_a(df_pklot_sampled, splits_dir)

    print("--- Generating Protocol C1 ---")
    protocol_dfs["Protocol C1"] = make_protocol_c1(df_pklot_sampled, df_cnr_sampled, splits_dir)

    print("--- Generating Protocol C2 ---")
    protocol_dfs["Protocol C2"] = make_protocol_c2(df_cnr_sampled, df_pklot_sampled, splits_dir)

    print("--- Generating Protocol D ---")
    protocol_dfs["Protocol D"] = make_protocol_d(df_pklot_sampled, df_cnr_sampled, splits_dir)

    print("--- Generating Split Summary ---")
    generate_split_summary(protocol_dfs, results_dir)

if __name__ == "__main__":
    main()
