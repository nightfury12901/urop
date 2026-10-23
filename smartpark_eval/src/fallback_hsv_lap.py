import cv2
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score

def extract_fallback_features(img):
    """
    Extracts deterministic fallback features:
    - mu_S: Mean HSV Saturation in [0, 1]
    - mu_V: Mean HSV Luminance/Value in [0, 1]
    - sigma_L2: Variance of Laplacian of grayscale (texture sharpness)
    img: float32 in [0, 1] or uint8 in [0, 255], RGB shape (H, W, 3)
    """
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            else:
                u8 = np.clip(img, 0, 255).astype(np.uint8)
        else:
            u8 = img
    else:
        raise ValueError(f"Expected numpy ndarray, got {type(img)}")

    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    mu_S = float(np.mean(hsv[:, :, 1])) / 255.0
    mu_V = float(np.mean(hsv[:, :, 2])) / 255.0

    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sigma_L2 = float(lap.var())

    return mu_S, mu_V, sigma_L2

def extract_features_batch(images):
    """Extracts features for a batch or list of images."""
    features = np.empty((len(images), 3), dtype=np.float32)
    for i, img in enumerate(images):
        features[i] = extract_fallback_features(img)
    return features

def tune_fallback_thresholds(val_images, val_labels, n_grid=12):
    """
    Grid-search optimization on validation data ONLY.
    Tunes:
    - tau_L: Laplacian variance threshold
    - tau_S: HSV saturation threshold
    - tau_V: HSV luminance deviation threshold
    - mu_V_asphalt: Baseline empty stall asphalt luminance
    """
    val_labels = np.asarray(val_labels).astype(int)
    feats = extract_features_batch(val_images)
    mu_S_vals = feats[:, 0]
    mu_V_vals = feats[:, 1]
    sigma_L2_vals = feats[:, 2]

    # Baseline asphalt luminance estimated from empty (free = 0) stalls
    free_mask = (val_labels == 0)
    if np.any(free_mask):
        mu_V_asphalt = float(np.median(mu_V_vals[free_mask]))
    else:
        mu_V_asphalt = float(np.median(mu_V_vals))

    dev_V_vals = np.abs(mu_V_vals - mu_V_asphalt)

    # Grid search candidate thresholds
    tau_L_cands = np.quantile(sigma_L2_vals, np.linspace(0.15, 0.85, n_grid))
    tau_S_cands = np.quantile(mu_S_vals, np.linspace(0.15, 0.85, n_grid))
    tau_V_cands = np.quantile(dev_V_vals, np.linspace(0.15, 0.85, n_grid))

    best_score = -1.0
    best_params = {
        "tau_L": float(tau_L_cands[len(tau_L_cands) // 2]),
        "tau_S": float(tau_S_cands[len(tau_S_cands) // 2]),
        "tau_V": float(tau_V_cands[len(tau_V_cands) // 2]),
        "mu_V_asphalt": float(mu_V_asphalt)
    }

    # Evaluate combinations on validation
    for t_L in tau_L_cands:
        l_pred = (sigma_L2_vals > t_L)
        for t_S in tau_S_cands:
            s_pred = (mu_S_vals > t_S)
            for t_V in tau_V_cands:
                v_pred = (dev_V_vals > t_V)
                # Prediction: occupied if high texture OR high saturation OR high luminance deviation
                pred = (l_pred | s_pred | v_pred).astype(int)
                score = balanced_accuracy_score(val_labels, pred)
                if score > best_score:
                    best_score = score
                    best_params = {
                        "tau_L": float(t_L),
                        "tau_S": float(t_S),
                        "tau_V": float(t_V),
                        "mu_V_asphalt": float(mu_V_asphalt)
                    }

    print(f"Fallback tuning best val balanced acc: {best_score:.4f} with params: {best_params}")
    return best_params

def predict_fallback(images, params):
    """
    Applies the tuned fallback heuristic to predict stall occupancy.
    Returns:
    - preds: binary array (0 = free, 1 = occupied)
    - scores: continuous confidence indicator in [0, 1]
    """
    feats = extract_features_batch(images)
    mu_S = feats[:, 0]
    mu_V = feats[:, 1]
    sigma_L2 = feats[:, 2]

    dev_V = np.abs(mu_V - params["mu_V_asphalt"])

    l_cond = sigma_L2 > params["tau_L"]
    s_cond = mu_S > params["tau_S"]
    v_cond = dev_V > params["tau_V"]

    preds = (l_cond | s_cond | v_cond).astype(int)

    # Heuristic score for ROC/PR curves
    norm_L = np.clip(sigma_L2 / (params["tau_L"] * 2.0 + 1e-6), 0.0, 1.0)
    norm_S = np.clip(mu_S / (params["tau_S"] * 2.0 + 1e-6), 0.0, 1.0)
    norm_V = np.clip(dev_V / (params["tau_V"] * 2.0 + 1e-6), 0.0, 1.0)
    scores = np.maximum.reduce([norm_L, norm_S, norm_V])

    return preds, scores
