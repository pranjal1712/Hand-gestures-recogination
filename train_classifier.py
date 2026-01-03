# train_classifier.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

DATA_FILE = "asl_landmarks.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Run merge_dataset.py first.")

    df = pd.read_csv(DATA_FILE)

    # Feature columns v0..v62
    feature_cols = [c for c in df.columns if c.startswith("v")]
    X = df[feature_cols].values.astype("float32")
    y_labels = df["label"].values

    print("Samples:", X.shape[0], "Features:", X.shape[1])
    return X, y_labels

def build_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    X, y_labels = load_data()

    # Encode string labels -> int
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    classes = le.classes_
    num_classes = len(classes)

    print("Classes:", classes)
    np.save(os.path.join(MODEL_DIR, "label_map.npy"), classes)
    print("Saved label map to", os.path.join(MODEL_DIR, "label_map.npy"))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(input_dim=X.shape[1], num_classes=num_classes)

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es]
    )

    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    print(f"Final train acc: {train_acc:.4f}")
    print(f"Final val acc:   {val_acc:.4f}")

    # Save Keras model
    h5_path = os.path.join(MODEL_DIR, "asl_landmark_model.h5")
    model.save(h5_path)
    print("Saved Keras model to", h5_path)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = os.path.join(MODEL_DIR, "asl_landmark_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite model to", tflite_path)

if __name__ == "__main__":
    main()
