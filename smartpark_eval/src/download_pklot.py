import datetime
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import pandas as pd
import requests
import yaml

def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def update_manifest(manifest_path, entry_name, entry_data):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
    manifest[entry_name] = entry_data
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest updated at {manifest_path}")

def main():
    with open("smartpark_eval/configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["paths"]["data_dir"]
    manifest_path = os.path.join(cfg["paths"]["results_dir"], "data_manifest.json")
    pklot_cfg = cfg["datasets"]["pklot"]
    target_path = os.path.join(data_dir, pklot_cfg["archive_name"])
    extract_dir = pklot_cfg["extract_dir"]
    url = pklot_cfg["url"]

    # 1. Download
    if os.path.exists(target_path):
        size = os.path.getsize(target_path)
        print(f"File {target_path} already exists ({size} bytes). Computing SHA-256...")
        sha = compute_sha256(target_path)
    else:
        temp_path = target_path + ".tmp"
        print(f"Downloading PKLot from {url} -> {target_path}...")
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, stream=True, headers=headers, timeout=30)
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        downloaded = 0
        t0 = time.time()
        last_log = t0
        h = hashlib.sha256()
        with open(temp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    h.update(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_log > 10:
                        speed = (downloaded / (1024 * 1024)) / (now - t0 + 1e-6)
                        pct = (downloaded / total_size * 100) if total_size > 0 else 0
                        print(f"  PKLot: {downloaded / (1024*1024):.1f}/{total_size/(1024*1024):.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s")
                        last_log = now
        shutil.move(temp_path, target_path)
        sha = h.hexdigest()
        size = downloaded
        print(f"PKLot download complete: {target_path} ({size} bytes, SHA-256: {sha})")

    manifest_entry = {
        "url": url,
        "download_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "size_bytes": size,
        "sha256": sha,
        "target_path": os.path.abspath(target_path)
    }
    update_manifest(manifest_path, "PKLot", manifest_entry)

    # 2. Extract sampled members
    sampled_pklot_csv = "smartpark_eval/splits/sampled_pklot.csv"
    if os.path.exists(sampled_pklot_csv):
        df_sampled = pd.read_csv(sampled_pklot_csv)
        needed_members = set(df_sampled["tar_member"])
        print(f"Extracting {len(needed_members)} sampled members into {extract_dir}...")
    else:
        needed_members = None
        print(f"Extracting full archive into {extract_dir}...")

    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(target_path, "r:*") as t:
        count = 0
        t0 = time.time()
        last_log = t0
        for member in t:
            if needed_members is None or member.name in needed_members:
                t.extract(member, path=extract_dir)
                count += 1
                now = time.time()
                if now - last_log > 10:
                    print(f"  Extracted {count}/{len(needed_members) if needed_members else '?'} files...")
                    last_log = now
        print(f"Extraction complete: {count} files extracted in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()
