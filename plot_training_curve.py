# plot_training_curve.py
"""
Plot training accuracy & loss curve from saved history (if exists).
If no history file found, optionally retrain model for a few epochs to generate curves.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

MODEL_H5 = Path("model/asl_landmark_model.h5")
HISTORY_NPY = Path("model/training_history.npy")
HISTORY_PKL = Path("model/training_history.pkl")
OUT = Path("analysis_outputs")
OUT.mkdir(exist_ok=True, parents=True)

# ---- Change for quick retrain ----
RETRAIN_IF_NO_HISTORY = True   # <-- set True to retrain quickly when no saved history
RETRAIN_EPOCHS = 10            # keep small for quick results (increase if you want nicer curves)
BATCH_SIZE = 32

def load_history():
    if HISTORY_PKL.exists():
        with open(HISTORY_PKL, "rb") as f:
            return pickle.load(f)
    if HISTORY_NPY.exists():
        return np.load(HISTORY_NPY, allow_pickle=True).item()
    # try json
    hjson = Path("model/training_history.json")
    if hjson.exists():
        return json.loads(hjson.read_text(encoding="utf8"))
    return None

hist = load_history()
if hist is None and RETRAIN_IF_NO_HISTORY:
    print("No saved history found; retraining model briefly to generate curves.")
    # quick retrain path: requires dataset (asl_landmarks.csv)
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import tensorflow as tf

    CSV_PATH = Path("asl_landmarks.csv")
    if not CSV_PATH.exists():
        raise FileNotFoundError("No training history and dataset not found to retrain.")
    df = pd.read_csv(CSV_PATH)
    # pick label last column logic like analyze_model
    label_col = None
    possible_label_cols = ["label","Label","target","class","y"]
    for c in df.columns:
        if str(c).lower() in possible_label_cols:
            label_col = c; break
    if label_col is None:
        last_col = df.columns[-1]
        sample = df[last_col].iloc[0]
        if isinstance(sample, str):
            label_col = last_col
    if label_col is None:
        raise ValueError("Could not detect label column for retrain.")

    X = df[[c for c in df.columns if c!=label_col]].values.astype("float32")
    y_raw = df[label_col].values
    # encode y if strings
    if y_raw.dtype.kind in ("U","S","O"):
        unique = sorted(set([str(x) for x in y_raw]))
        mapping = {v:i for i,v in enumerate(unique)}
        y = np.array([mapping[str(x)] for x in y_raw])
    else:
        y = y_raw.astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # load model
    model = tf.keras.models.load_model(str(MODEL_H5))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist_obj = model.fit(X_train, y_train, validation_data=(X_val,y_val),
                        epochs=RETRAIN_EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    hist = hist_obj.history
    # save history
    import pickle
    with open(HISTORY_PKL, "wb") as f:
        pickle.dump(hist, f)
    print("Retrain complete; saved history to", HISTORY_PKL)

if hist is None:
    print("No training history found. Set RETRAIN_IF_NO_HISTORY=True to retrain a short time.")
else:
    # hist expected keys: loss, accuracy, val_loss, val_accuracy OR acc, val_acc
    acc_key = "accuracy" if "accuracy" in hist else ("acc" if "acc" in hist else None)
    val_acc_key = "val_" + acc_key if acc_key else None
    loss_key = "loss"
    val_loss_key = "val_loss"

    epochs = range(1, len(hist[loss_key]) + 1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    if acc_key and val_acc_key and acc_key in hist:
        plt.plot(epochs, hist[acc_key], label="train_acc")
        if val_acc_key in hist: plt.plot(epochs, hist[val_acc_key], label="val_acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, hist[loss_key], label="train_loss")
    if val_loss_key in hist and val_loss_key in hist:
        plt.plot(epochs, hist[val_loss_key], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(OUT / "training_curve.png", dpi=200)
    print("Saved training_curve.png to", OUT / "training_curve.png")
