"""
Parking slot occupancy classifier (FREE vs OCCUPIED) - small CNN, ESP32-friendly.

Usage:
  python parking_ml.py                      # demo with synthetic images (works instantly)
  python parking_ml.py --data dataset       # use YOUR photos: dataset/free/*.jpg, dataset/occupied/*.jpg
  python parking_ml.py --data dataset --epochs 20
  python parking_ml.py --predict some_image.jpg     # test one image with the saved model
  python parking_ml.py --webcam                     # live demo with laptop webcam

Install:  pip install tensorflow scikit-learn pillow matplotlib opencv-python
"""
import argparse, os, glob
import numpy as np
from PIL import Image

IMG = 64  # 64x64 grayscale -> tiny model, fits ESP32
CLASSES = ["free", "occupied"]


# ---------- data ----------
def load_folder(root):
    X, y = [], []
    for label, name in enumerate(CLASSES):
        files = []
        for ext in ("jpg", "jpeg", "png", "bmp"):
            files += glob.glob(os.path.join(root, name, f"*.{ext}"))
        for f in files:
            img = Image.open(f).convert("L").resize((IMG, IMG))
            X.append(np.asarray(img, dtype=np.float32) / 255.0)
            y.append(label)
        print(f"  {name}: {len(files)} images")
    return np.array(X)[..., None], np.array(y)


def make_synthetic(n=1500, seed=0):
    """Fake top-down slot images: asphalt + lines (free) or + a car-like block (occupied)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n):
        occ = i % 2
        bg = rng.uniform(0.15, 0.6)                       # lighting level
        img = np.full((IMG, IMG), bg) + rng.normal(0, 0.05, (IMG, IMG))
        img[:, 2:4] = bg + 0.35                           # slot lines
        img[:, -4:-2] = bg + 0.35
        if occ:
            w, h = rng.integers(28, 42), rng.integers(38, 56)
            x0 = rng.integers(8, IMG - w - 8); y0 = rng.integers(4, max(5, IMG - h - 4))
            car = rng.uniform(0.0, 1.0)
            img[y0:y0 + h, x0:x0 + w] = car
            img[y0 + h // 4:y0 + h // 2, x0 + 4:x0 + w - 4] = np.clip(car + 0.3, 0, 1) * 0.5  # windshield
        img = np.clip(img + rng.normal(0, 0.03, img.shape), 0, 1)
        X.append(img.astype(np.float32)); y.append(occ)
    return np.array(X)[..., None], np.array(y)


def augment(X, seed=0):
    """Lighting robustness: random brightness/contrast, noise, horizontal flip (training set only)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    b = rng.uniform(-0.2, 0.2, (n, 1, 1, 1)).astype(np.float32)
    c = rng.uniform(0.7, 1.3, (n, 1, 1, 1)).astype(np.float32)
    out = np.clip((X - 0.5) * c + 0.5 + b + rng.normal(0, 0.02, X.shape), 0, 1).astype(np.float32)
    flip = rng.random(n) < 0.5
    out[flip] = out[flip][:, :, ::-1, :]
    return out


# ---------- model ----------
def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    m = models.Sequential([
        layers.Input((IMG, IMG, 1)),
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(0.2, value_range=(0, 1)),          # lighting robustness
        layers.RandomContrast(0.3),
        layers.Conv2D(8, 3, activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(16, 3, activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),  # 0 = free, 1 = occupied
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m


# ---------- train / evaluate ----------
def train(args):
    import tensorflow as tf
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    if args.data:
        print("Loading your dataset:"); X, y = load_folder(args.data)
    else:
        print("No --data given: using SYNTHETIC images (pipeline demo only)"); X, y = make_synthetic()

    # 70 / 15 / 15 split
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=1)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=1)
    print(f"train={len(Xtr)} val={len(Xval)} test={len(Xte)}")

    Xtr = np.concatenate([Xtr, augment(Xtr)]); ytr = np.concatenate([ytr, ytr])  # 2x training data
    model = build_model(); model.summary()
    hist = model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=args.epochs, batch_size=32, verbose=2)

    # test-set results (images the model has NEVER seen)
    prob = model.predict(Xte, verbose=0).ravel()
    pred = (prob > 0.5).astype(int)
    print("\n=== TEST SET RESULTS ===")
    print(classification_report(yte, pred, target_names=CLASSES, digits=3))
    cm = confusion_matrix(yte, pred)
    print("Confusion matrix:\n", cm)

    # plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(hist.history["accuracy"], label="train"); ax[0].plot(hist.history["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].legend()
    ax[1].plot(hist.history["loss"], label="train"); ax[1].plot(hist.history["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].legend()
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax[2], colorbar=False)
    ax[2].set_title("Confusion matrix (test)")
    plt.tight_layout(); plt.savefig("results.png", dpi=120)
    print("Saved results.png")

    # save models
    model.save("parking_model.keras")

    # int8 TFLite for ESP32
    def rep():
        for i in range(min(200, len(Xtr))):
            yield [Xtr[i:i + 1]]
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()
    open("parking_model_int8.tflite", "wb").write(tfl)
    fp32_kb = os.path.getsize("parking_model.keras") / 1024
    print(f"\nESP32 model: parking_model_int8.tflite = {len(tfl)/1024:.1f} KB  (Keras file: {fp32_kb:.0f} KB)")

    # accuracy of the int8 model on the test set (proves quantization didn't break it)
    interp = tf.lite.Interpreter(model_content=tfl); interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    s, z = inp["quantization"]; so, zo = out["quantization"]
    correct = 0
    for i in range(len(Xte)):
        q = np.round(Xte[i:i + 1] / s + z).astype(np.int8)
        interp.set_tensor(inp["index"], q); interp.invoke()
        p = (float(interp.get_tensor(out["index"])[0][0]) - zo) * so
        correct += int((p > 0.5) == yte[i])
    print(f"int8 TFLite test accuracy: {correct/len(Xte)*100:.1f}%  |  float model: {(pred==yte).mean()*100:.1f}%")
    print("Next step: xxd -i parking_model_int8.tflite > model.h  (for ESP32 TFLite Micro)")


# ---------- inference ----------
def predict_array(model, gray64):
    p = float(model.predict(gray64[None, ..., None], verbose=0)[0][0])
    return ("OCCUPIED" if p > 0.5 else "FREE"), (p if p > 0.5 else 1 - p)


def predict_image(path):
    import tensorflow as tf
    model = tf.keras.models.load_model("parking_model.keras")
    img = np.asarray(Image.open(path).convert("L").resize((IMG, IMG)), dtype=np.float32) / 255.0
    label, conf = predict_array(model, img)
    print(f"{path}: {label} ({conf*100:.1f}% confident)")


def webcam():
    import cv2, tensorflow as tf, time
    model = tf.keras.models.load_model("parking_model.keras")
    cap = cv2.VideoCapture(0)
    print("Press q to quit. The green box in the middle is the 'parking slot'.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = w // 2 - 120, h // 2 - 120, w // 2 + 120, h // 2 + 120
        crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (IMG, IMG)).astype(np.float32) / 255.0
        t = time.time(); label, conf = predict_array(model, crop); ms = (time.time() - t) * 1000
        color = (0, 0, 255) if label == "OCCUPIED" else (0, 200, 0)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.putText(frame, f"{label} {conf*100:.0f}%  {ms:.0f}ms", (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Parking slot ML", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="folder with free/ and occupied/ subfolders")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--predict", help="path of one image to classify")
    ap.add_argument("--webcam", action="store_true")
    a = ap.parse_args()
    if a.predict: predict_image(a.predict)
    elif a.webcam: webcam()
    else: train(a)