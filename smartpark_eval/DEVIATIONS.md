# Engineering and Protocol Deviations Log (DEVIATIONS.md)

This document records all deviations, adaptations, and operational adjustments made during the rebuilding and evaluation of the SmartPark CNN on public datasets (ODbL / CC BY 4.0), adhering strictly to Rule 7.

---

### 1. CNRPark-EXT Hosting Relocation
- **Documented Location:** `http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip` and `http://cnrpark.it/dataset/splits.zip`.
- **Observed Behavior:** HTTP GET requests to `cnrpark.it/dataset/*` return HTTP 404 (Not Found).
- **Adaptation:** Inspection of the official `cnrpark.it` homepage source revealed that dataset archives have been relocated to the official GitHub repository release asset storage:
  - `CNR-EXT-Patches-150x150.zip`: `https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT-Patches-150x150.zip` (449,502,403 bytes, SHA-256: `84169527af6ad6aaf51729663b59283bc5f28368ff9ddd7856198eec226056d6`)
  - `splits.zip`: `https://github.com/fabiocarrara/deep-parking/releases/download/archive/splits.zip` (27,249,817 bytes, SHA-256: `77910fa59759833d6e1f236a8e4f785884997bb12d883ecce3f3fef16cb51ef6`)
- **Impact:** None. Dataset contents, labels, and file trees are 100% identical to the original benchmark published by Amato et al. (2017). Recorded in `results/data_manifest.json`.

---

### 2. PKLot Archive Handling and Selective Extraction
- **Documented Location:** `http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz` (~4.6 GB).
- **Observed Behavior:** The official UFPR host serves `PKLot.tar.gz` (4,898,276,304 bytes), containing 695,898 individual segmented patch JPEG files. On Windows NTFS filesystems, extracting ~700,000 separate files creates severe MFT table fragmentation and disk thrashing.
- **Adaptation:**
  - The bit-identical archive was downloaded from Hugging Face (`teenygrad/pklot/resolve/main/PKLot.tar.gz`, verified bit-for-bit against official UFPR checksum).
  - In accordance with Section 3.3 (stratified sampling of $N_{\text{per\_group}} = 300$ patches per class per (camera, date) group, stratified across hours of the day), the exact 50,952 patches required across all protocols were indexed first.
  - Extraction from the archive selectively targeted these 50,952 patches directly into `smartpark_eval/data/pklot/PKLot/PKLotSegmented/`, reducing disk footprint while guaranteeing exact adherence to Protocols A, C1, C2, and D.
- **Impact:** Eliminates disk thrashing while preserving exact, leak-free evaluation partitions.

---

### 3. Resolution of Paper Architectural Inconsistencies (M0 vs M1)
- **Documented Issue in Paper:**
  - The paper claims that an 8x8x64 tensor flattens to 2,304, which is mathematically false ($8 \times 8 \times 64 = 4,096$).
  - 2,304 corresponds to $6 \times 6 \times 64$, which would occur only if unpadded (valid) convolutions were used ($64 \rightarrow 62 \rightarrow 31 \rightarrow 29 \rightarrow 14 \rightarrow 12 \rightarrow 6$).
  - Furthermore, the stated parameter count of 24,577 does not match an 8x8x64 Flatten + Dense(64) layer (which has over 262k parameters in the Dense layer alone).
- **Adaptation:**
  - **M0 (Paper Baseline):** Built strictly per the text specification: Conv(16)-Pool, Conv(32)-Pool, Conv(64)-Pool (all 3x3 ReLU, same padding), Flatten ($8 \times 8 \times 64 = 4,096$), Dense(64) ReLU, Dropout(0.5), Dense(1, sigmoid). Real parameter count: **285,857 parameters** (1.09 MB FP32).
  - **M1 (Proposed Compact Model):** Solves the parameter explosion by replacing Flatten + Dense(64) with `GlobalAveragePooling2D()` and adding `BatchNormalization()`. With filter widths (16, 32, 64), M1 contains **24,097 parameters** (94.13 KB FP32), fitting well within ESP32 SRAM constraints.

---

### 4. Evidential Deep Learning (M2) Head
- **Specification:** M2 evidential head (Dirichlet prior loss) is optional if time permits.
- **Decision:** As specified in Section 6 and Section 12, M2 was skipped and marked as "unvalidated" in `ablation.csv` and `CLAIMS_TO_UPDATE.md`. Focus was directed to rigorous evaluation of M0, M1 (FP32), M1 (INT8), and the deterministic fallback on all four protocols.
