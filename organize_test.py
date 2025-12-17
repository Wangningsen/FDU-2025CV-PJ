#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import shutil

# 路径配置
CSV_PATH = "label.csv"
SRC_DIR = "/data1/nwang60/datasets/CV-Dataset/test"
DST_ROOT = "/data1/nwang60/datasets/CV-Dataset/test_new"

CLASS_DIRS = {
    "0": "0_real",
    "1": "1_fake",
}

def main():
    # 创建目标子文件夹
    for lbl, sub in CLASS_DIRS.items():
        os.makedirs(os.path.join(DST_ROOT, sub), exist_ok=True)

    num_total = 0
    num_missing = 0

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row["image_id"]
            label = str(row["label"]).strip()

            if label not in CLASS_DIRS:
                print(f"[WARN] Unknown label {label} for image {img_name}, skip")
                continue

            src_path = os.path.join(SRC_DIR, img_name)
            if not os.path.isfile(src_path):
                print(f"[WARN] File not found: {src_path}")
                num_missing += 1
                continue

            dst_dir = os.path.join(DST_ROOT, CLASS_DIRS[label])
            dst_path = os.path.join(dst_dir, img_name)

            shutil.copy2(src_path, dst_path)
            num_total += 1

    print("Done.")
    print(f"Copied images: {num_total}")
    print(f"Missing images: {num_missing}")

if __name__ == "__main__":
    main()
