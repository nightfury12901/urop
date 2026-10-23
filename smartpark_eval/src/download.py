import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
import requests
import yaml

def load_config(config_path="smartpark_eval/configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def download_file(url, target_path, num_retries=3):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        size = os.path.getsize(target_path)
        print(f"File {target_path} already exists ({size} bytes). Computing SHA-256...")
        sha = compute_sha256(target_path)
        return {
            "url": url,
            "download_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "size_bytes": size,
            "sha256": sha,
            "target_path": os.path.abspath(target_path)
        }

    temp_path = target_path + ".tmp"
    print(f"Downloading {url} -> {target_path}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for attempt in range(num_retries):
        try:
            r = requests.get(url, stream=True, headers=headers, timeout=30)
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            downloaded = 0
            t0 = time.time()
            last_log = t0
            
            h = hashlib.sha256()
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        h.update(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_log > 5:
                            speed = (downloaded / (1024 * 1024)) / (now - t0 + 1e-6)
                            pct = (downloaded / total_size * 100) if total_size > 0 else 0
                            print(f"  {downloaded / (1024*1024):.1f}/{total_size/(1024*1024):.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s")
                            last_log = now
                            
            shutil.move(temp_path, target_path)
            sha = h.hexdigest()
            print(f"Download complete: {target_path} ({downloaded} bytes, SHA-256: {sha})")
            return {
                "url": url,
                "download_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "size_bytes": downloaded,
                "sha256": sha,
                "target_path": os.path.abspath(target_path)
            }
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if attempt == num_retries - 1:
                raise

def extract_zip(archive_path, extract_dir):
    print(f"Extracting {archive_path} -> {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as z:
        z.extractall(extract_dir)
    print(f"Extraction of {archive_path} complete.")

def extract_tar(archive_path, extract_dir):
    print(f"Extracting {archive_path} -> {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as t:
        t.extractall(extract_dir)
    print(f"Extraction of {archive_path} complete.")

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
    parser = argparse.ArgumentParser(description="Download and verify datasets.")
    parser.add_argument("--config", default="smartpark_eval/configs/default.yaml", help="Path to config file.")
    parser.add_argument("--dataset", choices=["cnr_ext", "pklot", "all"], default="cnr_ext", help="Dataset to download.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg["paths"]["data_dir"]
    manifest_path = os.path.join(cfg["paths"]["results_dir"], "data_manifest.json")

    if args.dataset in ["cnr_ext", "all"]:
        cnr_cfg = cfg["datasets"]["cnr_ext"]
        archive_path = os.path.join(data_dir, cnr_cfg["archive_name"])
        manifest_entry = download_file(cnr_cfg["url"], archive_path)
        update_manifest(manifest_path, "CNR-EXT-Patches-150x150", manifest_entry)
        
        splits_path = os.path.join(data_dir, cnr_cfg["splits_archive_name"])
        splits_entry = download_file(cnr_cfg["splits_url"], splits_path)
        update_manifest(manifest_path, "CNR-EXT-splits", splits_entry)

        extract_dir = cnr_cfg["extract_dir"]
        if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) == 0:
            extract_zip(archive_path, extract_dir)
        splits_extract_dir = os.path.join(extract_dir, "splits")
        if not os.path.exists(splits_extract_dir):
            extract_zip(splits_path, splits_extract_dir)

    if args.dataset in ["pklot", "all"]:
        pklot_cfg = cfg["datasets"]["pklot"]
        archive_path = os.path.join(data_dir, pklot_cfg["archive_name"])
        try:
            manifest_entry = download_file(pklot_cfg["url"], archive_path)
        except Exception as e:
            print(f"Failed primary PKLot url, trying fallback: {e}")
            manifest_entry = download_file(pklot_cfg["fallback_url"], archive_path)
        update_manifest(manifest_path, "PKLot", manifest_entry)

        extract_dir = pklot_cfg["extract_dir"]
        if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) == 0:
            extract_tar(archive_path, extract_dir)

if __name__ == "__main__":
    main()
