# model_summary.py
"""
Print and save layer-by-layer model architecture summary,
parameter counts, optimizer settings if available.
"""

import tensorflow as tf
from pathlib import Path
import numpy as np
import json

MODEL_PATH = Path("model/asl_landmark_model.h5")
OUT = Path("analysis_outputs")
OUT.mkdir(exist_ok=True, parents=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

model = tf.keras.models.load_model(str(MODEL_PATH))
model.summary()

# Save textual summary
from io import StringIO
stream = StringIO()
model.summary(print_fn=lambda s: stream.write(s + "\n"))
summary_text = stream.getvalue()
(OUT / "model_summary.txt").write_text(summary_text, encoding="utf8")
print("Saved model_summary.txt")

# Layer-by-layer table
lines = []
lines.append("layer_idx,layer_name,layer_type,output_shape,trainable,params")
for i, layer in enumerate(model.layers):
    name = layer.name
    ltype = layer.__class__.__name__
    out_shape = getattr(layer, "output_shape", "unknown")
    trainable = getattr(layer, "trainable", False)
    params = layer.count_params()
    lines.append(f"{i},{name},{ltype},{out_shape},{trainable},{params}")

(OUT / "model_layers_table.csv").write_text("\n".join(lines), encoding="utf8")
print("Saved model_layers_table.csv")

# total params
total_params = model.count_params()
text = f"Total parameters: {total_params}\n"
# optimizer, loss, lr - if model compiled with these saved
opt_info = "N/A"
if hasattr(model, "optimizer") and model.optimizer is not None:
    try:
        opt_name = model.optimizer._name if hasattr(model.optimizer, "_name") else str(model.optimizer)
        lr = getattr(model.optimizer, "learning_rate", None)
        text += f"Optimizer: {opt_name}\n"
        if lr is not None:
            try:
                lr_val = float(tf.keras.backend.get_value(lr))
                text += f"Learning rate: {lr_val}\n"
            except Exception:
                text += f"Learning rate (object): {lr}\n"
    except Exception:
        pass

# Loss function (if compiled)
if hasattr(model, "loss") and model.loss is not None:asw-
    text += f"Loss function: {model.loss}\n"

(OUT / "model_architecture.txt").write_text(summary_text + "\n\n" + text, encoding="utf8")
print("Saved model_architecture.txt")
