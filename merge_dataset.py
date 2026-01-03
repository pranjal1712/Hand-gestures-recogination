# merge_dataset.py

import os
import csv

DATASET_DIR = "dataset"
OUT_FILE = "asl_landmarks.csv"

def main():
    rows = []

    # Har label ke folder se CSV uthao
    for label in sorted(os.listdir(DATASET_DIR)):
        label_path = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_path):
            continue

        print(f"Reading label: {label}")
        for file in os.listdir(label_path):
            if not file.endswith(".csv"):
                continue
            fpath = os.path.join(label_path, file)
            with open(fpath, "r") as f:
                reader = csv.reader(f)
                row = next(reader, None)
                if row is None:
                    continue
                # 63 features + label last
                rows.append(row + [label])

    # Header bana do: v0..v62 + label
    num_features = 63
    header = [f"v{i}" for i in range(num_features)] + ["label"]

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Saved merged dataset: {OUT_FILE}")
    print(f"Total samples: {len(rows)}")

if __name__ == "__main__":
    main()
