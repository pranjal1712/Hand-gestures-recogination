# analyze_model.py
"""
Load dataset + model, run inference on a held-out test set,
produce confusion matrix (image), classification report (text),
and sample terminal output showing predictions (top-5).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

# ============ CONFIG ============
MODEL_PATH = Path("model/asl_landmark_model.h5")   # adjust if needed
LABEL_NUMPY = Path("model/label_map.npy")         # optional
LABEL_TXT = Path("model/labels.txt")              # optional
CSV_PATH = Path("asl_landmarks.csv")              # your collected landmarks CSV
OUTPUT_DIR = Path("analysis_outputs")
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_K = 5
# ==============================

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# --- load labels ---
def load_labels():
    if LABEL_TXT.exists():
        labs = [l.strip() for l in LABEL_TXT.read_text(encoding="utf8").splitlines() if l.strip()]
        return labs
    if LABEL_NUMPY.exists():
        arr = np.load(LABEL_NUMPY, allow_pickle=True)
        return [str(x) for x in arr.tolist()]
    # fallback default A-Z (27 classes / adjust)
    return [chr(ord("A")+i) for i in range(26)] + ["del","nothing","space"]

labels = load_labels()
num_classes = len(labels)
print("Labels (len={}):".format(len(labels)), labels)

# --- load model ---
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(str(MODEL_PATH))
print("Model loaded.")

# --- load dataset CSV (robust) ---
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")

print("Loading CSV:", CSV_PATH)
# Try reading common formats: CSV with header, or first N columns are coords and last column label
df = pd.read_csv(CSV_PATH, header=0, low_memory=False)
print("CSV columns:", df.columns.tolist()[:10], "... (total {})".format(len(df.columns)))

# Try to detect label column:
label_col = None
possible_label_cols = ["label", "Label", "target", "class", "y"]
for c in df.columns:
    if str(c).lower() in [p.lower() for p in possible_label_cols]:
        label_col = c
        break

if label_col is None:
    # assume last column is label if non-numeric
    last_col = df.columns[-1]
    sample = df[last_col].iloc[0]
    if isinstance(sample, str) or not np.issubdtype(type(sample), np.number):
        label_col = last_col
    else:
        # else look for a col named '0'..'62' etc and no explicit label; maybe labels in separate file
        # If no label found, try to guess there's a 'label' column in index or similar.
        raise ValueError("Could not detect label column automatically. Please ensure your CSV has a label column named 'label' or similar.")
print("Detected label column:", label_col)

# build X, y
# assume feature columns are all numeric columns except label_col
feature_cols = [c for c in df.columns if c != label_col]
X = df[feature_cols].values.astype(np.float32)
y_raw = df[label_col].values

# If y_raw is string labels, map to indices according to 'labels' loaded above (or create mapping)
if y_raw.dtype.kind in ("U", "S", "O"):
    # ensure labels mapping contains all labels seen
    unique = sorted(set([str(x) for x in y_raw]))
    mapping = {lab: labels.index(lab) if lab in labels else None for lab in unique}
    # if some labels not in labels list, extend labels list
    added = False
    for u in unique:
        if mapping[u] is None:
            labels.append(u)
            mapping[u] = len(labels)-1
            added = True
    if added:
        print("Extended labels to include new classes. New labels:", labels)
    y = np.array([mapping[str(x)] for x in y_raw], dtype=np.int32)
else:
    y = y_raw.astype(int)

print("Dataset shape X:", X.shape, "y:", y.shape, "num_classes:", len(set(y)))

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print("Train/test split:", X_train.shape, X_test.shape)

# run predictions on test set in batches (and show top-k terminal output)
batch_size = 64
preds = model.predict(X_test, batch_size=batch_size)
if preds.ndim == 2:
    probs = preds
else:
    probs = preds.reshape(len(X_test), -1)

y_pred = np.argmax(probs, axis=1)

# classification report
report = classification_report(y_test, y_pred, target_names=[str(l) for l in labels], digits=4)
print("\n=== Classification Report ===\n")
print(report)
(OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf8")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.set(font_scale=0.8)
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.close()
print("Saved confusion matrix to", OUTPUT_DIR / "confusion_matrix.png")

# terminal output: sample predictions with top-K
def top_k_probs(prob_row, k=5):
    idx = np.argsort(prob_row)[::-1][:k]
    return [(i, prob_row[i]) for i in idx]

out_lines = []
n_sample = min(50, len(X_test))
out_lines.append("Sample predictions (first {} test samples):\n".format(n_sample))
for i in range(n_sample):
    row = X_test[i]
    true_idx = int(y_test[i])
    prob_row = probs[i]
    topk = top_k_probs(prob_row, k=TOP_K)
    topk_str = ", ".join([f"{labels[idx]}:{p:.4f}" for idx,p in topk])
    line = f"Sample {i}: True={labels[true_idx]} Pred={labels[y_pred[i]]} (conf={prob_row[y_pred[i]]:.4f}) Top-{TOP_K} -> {topk_str}"
    out_lines.append(line)
    # also print some to terminal
    if i < 10:
        print(line)

(OUTPUT_DIR / "predictions_sample.txt").write_text("\n".join(out_lines), encoding="utf8")
print("Saved sample predictions to", OUTPUT_DIR / "predictions_sample.txt")

# Save overall metrics summary
acc = np.mean(y_pred == y_test)
summary = f"Test accuracy: {acc:.4f}\nSamples test: {len(y_test)}\n"
(OUTPUT_DIR / "metrics_summary.txt").write_text(summary, encoding="utf8")
print(summary)
print("All outputs saved to", OUTPUT_DIR)
