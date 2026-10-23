import argparse
import os
import re
import numpy as np
import pandas as pd
import yaml

def load_config(config_path="smartpark_eval/configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def stratify_sample_group(df_group, n_per_group=300, seed=42):
    """
    Samples up to n_per_group items evenly across the day (stratified by hour).
    Ensures deterministic, even temporal coverage.
    """
    if len(df_group) <= n_per_group:
        return df_group

    df_group = df_group.sort_values(by="path").reset_index(drop=True)
    hours = sorted(df_group["hour"].unique())
    num_hours = len(hours)
    
    base_per_hour = n_per_group // num_hours
    remainder = n_per_group % num_hours

    selected_indices = []
    hour_to_indices = {h: df_group.index[df_group["hour"] == h].tolist() for h in hours}

    # First pass: take uniform slice per hour
    unused_quota = 0
    pool_for_second_pass = []

    for idx, h in enumerate(hours):
        quota = base_per_hour + (1 if idx < remainder else 0)
        h_indices = hour_to_indices[h]
        if len(h_indices) <= quota:
            selected_indices.extend(h_indices)
            unused_quota += (quota - len(h_indices))
        else:
            step_indices = np.linspace(0, len(h_indices) - 1, quota, dtype=int)
            taken = [h_indices[i] for i in step_indices]
            selected_indices.extend(taken)
            remaining_in_h = [idx_val for i, idx_val in enumerate(h_indices) if i not in set(step_indices)]
            pool_for_second_pass.extend(remaining_in_h)

    # Second pass: fill unused quota from pool
    if unused_quota > 0 and len(pool_for_second_pass) > 0:
        fill_count = min(unused_quota, len(pool_for_second_pass))
        fill_step = np.linspace(0, len(pool_for_second_pass) - 1, fill_count, dtype=int)
        for i in fill_step:
            selected_indices.append(pool_for_second_pass[i])

    selected_indices = sorted(list(set(selected_indices)))
    return df_group.loc[selected_indices]

def index_cnr_ext(cnr_dir, n_per_group=300):
    """
    Parses CNRPark-EXT metadata from LABELS/all.txt.
    """
    labels_file = os.path.join(cnr_dir, "LABELS", "all.txt")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Missing {labels_file}")

    print(f"Indexing CNRPark-EXT from {labels_file}...")
    records = []
    patches_base = os.path.join(cnr_dir, "PATCHES")

    with open(labels_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            rel_path, label_str = parts[0], parts[1]
            label = int(label_str)
            # Example: RAINY/2016-02-12/camera7/R_2016-02-12_12.25_C07_275.jpg
            tokens = rel_path.split("/")
            weather = tokens[0].upper()
            date = tokens[1]
            camera = tokens[2]
            fname = tokens[3]

            # Parse hour from filename: e.g. R_2016-02-12_12.25_C07_275.jpg
            m = re.search(r"(\d{2})\.(\d{2})", fname)
            hour = int(m.group(1)) if m else 12

            full_path = os.path.join(patches_base, weather, date, camera, fname)
            records.append({
                "path": full_path.replace("\\", "/"),
                "rel_path": rel_path,
                "label": label,
                "camera": camera,
                "date": date,
                "weather": weather,
                "hour": hour,
                "dataset": "CNRPark-EXT"
            })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} entries for CNRPark-EXT.")
    
    # Stratified sampling
    sampled_chunks = []
    for _, group in df.groupby(["camera", "date", "label"]):
        sampled_chunks.append(stratify_sample_group(group, n_per_group=n_per_group))
    df_sampled = pd.concat(sampled_chunks, ignore_index=True)
    print(f"Sampled {len(df_sampled)} entries for CNRPark-EXT (max {n_per_group} per camera/date/class).")
    return df, df_sampled

def index_pklot(pklot_list_file, pklot_extract_dir, n_per_group=300):
    """
    Parses PKLot metadata from splits/PKLot/all.txt.
    """
    if not os.path.exists(pklot_list_file):
        raise FileNotFoundError(f"Missing {pklot_list_file}")

    print(f"Indexing PKLot from {pklot_list_file}...")
    records = []
    with open(pklot_list_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            rel_path, label_str = parts[0], parts[1]
            label = int(label_str)
            # Example: UFPR04/Sunny/2012-12-25/Empty/2012-12-25_12_00_07#027.jpg
            tokens = rel_path.split("/")
            lot = tokens[0]  # UFPR04, UFPR05, PUC
            weather = tokens[1].upper()
            date = tokens[2]
            fname = tokens[4]

            # Parse hour from filename: e.g. 2012-12-25_12_00_07#027.jpg
            m = re.search(r"_(\d{2})_\d{2}_\d{2}", fname)
            hour = int(m.group(1)) if m else 12

            # Expected relative path inside tar or extracted folder
            full_path = os.path.join(pklot_extract_dir, "PKLot", "PKLotSegmented", rel_path)
            tar_member_path = f"PKLot/PKLotSegmented/{rel_path}"

            records.append({
                "path": full_path.replace("\\", "/"),
                "tar_member": tar_member_path,
                "rel_path": rel_path,
                "label": label,
                "camera": lot,  # lot acts as camera/lot
                "date": date,
                "weather": weather,
                "hour": hour,
                "dataset": "PKLot"
            })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} entries for PKLot.")

    # Stratified sampling
    sampled_chunks = []
    for _, group in df.groupby(["camera", "date", "label"]):
        sampled_chunks.append(stratify_sample_group(group, n_per_group=n_per_group))
    df_sampled = pd.concat(sampled_chunks, ignore_index=True)
    print(f"Sampled {len(df_sampled)} entries for PKLot (max {n_per_group} per camera/date/class).")
    return df, df_sampled

def main():
    parser = argparse.ArgumentParser(description="Build dataset metadata tables and sampled indices.")
    parser.add_argument("--config", default="smartpark_eval/configs/default.yaml", help="Path to config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits_dir = cfg["paths"]["splits_dir"]
    os.makedirs(splits_dir, exist_ok=True)
    n_per_group = cfg["sampling"]["n_per_group"]

    # 1. Index CNRPark-EXT
    cnr_dir = cfg["datasets"]["cnr_ext"]["extract_dir"]
    df_cnr_all, df_cnr_sampled = index_cnr_ext(cnr_dir, n_per_group=n_per_group)
    df_cnr_all.to_csv(os.path.join(splits_dir, "metadata_cnr_ext.csv"), index=False)
    df_cnr_sampled.to_csv(os.path.join(splits_dir, "sampled_cnr_ext.csv"), index=False)

    # 2. Index PKLot
    pklot_list_file = os.path.join(cnr_dir, "splits", "splits", "PKLot", "all.txt")
    pklot_extract_dir = cfg["datasets"]["pklot"]["extract_dir"]
    df_pklot_all, df_pklot_sampled = index_pklot(pklot_list_file, pklot_extract_dir, n_per_group=n_per_group)
    df_pklot_all.to_csv(os.path.join(splits_dir, "metadata_pklot.csv"), index=False)
    df_pklot_sampled.to_csv(os.path.join(splits_dir, "sampled_pklot.csv"), index=False)

    print("Metadata indexing and sampling complete!")

if __name__ == "__main__":
    main()
