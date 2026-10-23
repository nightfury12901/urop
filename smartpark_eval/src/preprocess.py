import cv2
import numpy as np
from PIL import Image

def prepare(img, degrade=False, rng=None, target_size=(64, 64)):
    """
    Single shared preprocessing + sensor degradation function.
    img: file path, PIL Image, or numpy array (H, W, 3) or (H, W)
    degrade: whether to emulate ESP32-CAM sensor degradation
    rng: np.random.Generator instance (or None to create a default one)
    target_size: (width, height) = (64, 64)
    Returns: np.ndarray of shape (64, 64, 3), dtype float32, in range [0, 1]
    """
    if rng is None:
        rng = np.random.default_rng()

    # Load / convert to uint8 RGB numpy array
    if isinstance(img, str):
        pil_img = Image.open(img).convert("RGB")
        img_arr = np.array(pil_img, dtype=np.uint8)
    elif isinstance(img, Image.Image):
        img_arr = np.array(img.convert("RGB"), dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img_arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            else:
                img_arr = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img_arr = img.copy()
        if img_arr.ndim == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        elif img_arr.ndim == 3 and img_arr.shape[2] == 1:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        elif img_arr.ndim == 3 and img_arr.shape[2] == 4:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")

    h_orig, w_orig = img_arr.shape[:2]

    # Sensor degradation (applied BEFORE final 64x64 resize)
    if degrade:
        # 1. Downscale to random short side of 24-48 px, then upscale back
        short_side = int(rng.integers(24, 49))
        if h_orig < w_orig:
            new_h = short_side
            new_w = max(1, int(round(w_orig * (short_side / h_orig))))
        else:
            new_w = short_side
            new_h = max(1, int(round(h_orig * (short_side / w_orig))))
        down = cv2.resize(img_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_arr = cv2.resize(down, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        # 2. JPEG re-encode at quality 10-30
        jpeg_q = int(rng.integers(10, 31))
        bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        success, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
        if success:
            dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            img_arr = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)

        # 3. Per-channel gain (0.9 - 1.1) for color cast
        gain = rng.uniform(0.9, 1.1, size=(1, 1, 3))
        img_float = np.clip(img_arr.astype(np.float32) * gain, 0.0, 255.0)

        # 4. Random gamma (0.7 - 1.4)
        gamma = float(rng.uniform(0.7, 1.4))
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.float32)
        lut = np.clip(lut, 0.0, 255.0).astype(np.uint8)
        img_u8 = cv2.LUT(np.clip(img_float, 0, 255).astype(np.uint8), lut)

        # 5. Gaussian noise (sigma 0.01 - 0.04 in [0, 1] range)
        noise_sigma = float(rng.uniform(0.01, 0.04)) * 255.0
        noise = rng.normal(0.0, noise_sigma, size=img_u8.shape)
        img_arr = np.clip(img_u8.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

    # Resize to target (64x64)
    target_w, target_h = target_size
    curr_h, curr_w = img_arr.shape[:2]
    if curr_h > target_h or curr_w > target_w:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR

    resized = cv2.resize(img_arr, (target_w, target_h), interpolation=interp)

    # Scale to [0, 1] (divide by 255)
    normalized = resized.astype(np.float32) / 255.0
    return normalized

def prepare_batch(images, degrade=False, rng=None, target_size=(64, 64)):
    """Helper to process a list or batch of images."""
    if rng is None:
        rng = np.random.default_rng()
    out = np.empty((len(images), target_size[1], target_size[0], 3), dtype=np.float32)
    for i, img in enumerate(images):
        out[i] = prepare(img, degrade=degrade, rng=rng, target_size=target_size)
    return out
