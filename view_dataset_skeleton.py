import os
import csv
import cv2
import numpy as np
from glob import glob

DATASET_DIR = "dataset"

# Mediapipe hand connections (hard-coded indices)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (0,5),(5,6),(6,7),(7,8),        # Index
    (0,9),(9,10),(10,11),(11,12),   # Middle
    (0,13),(13,14),(14,15),(15,16), # Ring
    (0,17),(17,18),(18,19),(19,20)  # Pinky
]

CANVAS_SIZE = 512

def load_sample(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        row = next(reader)
    vals = np.array(row, dtype=np.float32)  # 63
    pts = vals.reshape(21, 3)  # (21,3) -> x,y,z
    return pts

def draw_hand(pts):
    # pts : (21,3) normalized [0,1]
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)

    # convert normalized to pixel
    xs = (pts[:,0] * CANVAS_SIZE).astype(int)
    ys = (pts[:,1] * CANVAS_SIZE).astype(int)

    # draw connections
    for i,j in HAND_CONNECTIONS:
        cv2.line(canvas, (xs[i], ys[i]), (xs[j], ys[j]), (0,255,0), 2)

    # draw points
    for x,y in zip(xs,ys):
        cv2.circle(canvas, (x,y), 4, (255,255,255), -1)

    return canvas

def main():
    # sab CSV utha lo
    all_csv = []
    for letter in sorted(os.listdir(DATASET_DIR)):
        letter_path = os.path.join(DATASET_DIR, letter)
        if not os.path.isdir(letter_path):
            continue
        files = sorted(glob(os.path.join(letter_path, "*.csv")))
        for f in files:
            all_csv.append((letter, f))

    if not all_csv:
        print("No CSV files found in dataset/")
        return

    idx = 0
    print("Total samples:", len(all_csv))
    print("Controls: n = next, p = previous, q = quit")

    while True:
        label, path = all_csv[idx]
        pts = load_sample(path)
        img = draw_hand(pts)

        # text info
        info = f"{idx+1}/{len(all_csv)} | Label: {label} | {os.path.basename(path)}"
        cv2.putText(img, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Dataset Skeleton Viewer", img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('n'):  # next
            idx = (idx + 1) % len(all_csv)
        elif key == ord('p'):  # previous
            idx = (idx - 1) % len(all_csv)
        elif key == ord('q'):  # quit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
