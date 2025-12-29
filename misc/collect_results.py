#!/usr/bin/env python
"""
Collect eval_val.py metrics from weights/*/*.csv into:
  1) a global summary CSV
  2) several grouped CSVs (single-branch, arch ablation, regularization, leaderboard)
"""

import os
import csv

WEIGHTS_DIR = "weights"
RESULTS_DIR = "results"

# 统一列顺序
FIELD_ORDER = [
    "experiment",
    "model_size",
    "two_branch",
    "channel_mode",
    "fft_highpass_only",
    "fft_low_cut_ratio",
    "use_tta",
    "num_samples",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tp",
    "tn",
    "fp",
    "fn",
    "pos_rate",
    "val_ratio",
    "seed",
    "checkpoint",
]


def load_metrics_csv(csv_path):
    """
    读取 eval_val.py 生成的单个 metrics csv (两列: metric,value) 为 dict。
    """
    metrics = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # ["metric","value"]
        for row in reader:
            if not row or len(row) < 2:
                continue
            key, val = row[0], row[1]
            metrics[key] = val
    return metrics


def collect_all_results():
    """
    扫描 weights 下所有子目录, 读取 <exp_name>.csv, 汇总成一张总表.
    返回: list[dict], 每个 dict 是一行.
    """
    rows = []
    if not os.path.isdir(WEIGHTS_DIR):
        print(f"weights dir '{WEIGHTS_DIR}' not found, nothing to do.")
        return rows

    for exp_name in sorted(os.listdir(WEIGHTS_DIR)):
        exp_dir = os.path.join(WEIGHTS_DIR, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        csv_path = os.path.join(exp_dir, f"{exp_name}.csv")
        if not os.path.isfile(csv_path):
            # 有些权重目录可能还没 eval 或命名不一致
            print(f"[WARN] metrics csv not found for experiment '{exp_name}': {csv_path}")
            continue

        metrics = load_metrics_csv(csv_path)
        row = {k: "" for k in FIELD_ORDER}
        row["experiment"] = exp_name

        # 把 metrics 里有的字段写进去
        for key, val in metrics.items():
            if key in row:
                row[key] = val

        rows.append(row)

    return rows


def write_csv(path, rows, fieldnames=None):
    if not rows:
        print(f"[INFO] No rows for {path}, skip.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = FIELD_ORDER
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[OK] Wrote {len(rows)} rows to {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. 汇总所有实验
    all_rows = collect_all_results()
    if not all_rows:
        return

    # 建 index: name -> row
    row_by_name = {r["experiment"]: r for r in all_rows}

    # 1) 全量结果
    all_path = os.path.join(RESULTS_DIR, "all_results.csv")
    write_csv(all_path, all_rows)

    # 定义几个 group, 用我们前面讨论的实验名
    groups = {
        # 单塔 small, 通道与轻量正则消融
        "single_branch_small": [
            "arch_small_rgb",
            "arch_small_rgb_grad",
            "arch_small_rgb_fft",
            "arch_small_rgb_fft_grad_full",
            "arch_small_rgb_fft_grad_hp01",
            "reg_single_small_rgb_grad_ls",
            "reg_single_small_rgb_grad_ls_mixup",
        ],
        # 架构和尺度 ablation (rgb_grad 为主)
        "arch_ablation": [
            "arch_small_rgb_grad",
            "arch_large_rgb_grad",
            "arch_twobranch_small_rgb_grad",
            "arch_twobranch_large_rgb_grad",
            "extra_twobranch_small_lr5e4",
            "extra_twobranch_large_lr5e4",
            "extra_twobranch_large_lr3e4",
        ],
        # small two-branch 上的正则化消融
        "reg_small_twobranch": [
            "arch_twobranch_small_rgb_grad",
            "reg_twobranch_small_strong",
            "reg_twobranch_small_strong_ls",
            "reg_twobranch_small_strong_mixup",
            "reg_twobranch_small_strong_ls_mixup",
            "extra_twobranch_small_lr5e4",
            "extra_twobranch_small_lr3e4",
            "extra_twobranch_small_ls_only",
            "extra_twobranch_small_mixup_only",
            "extra_twobranch_small_strong_ls_mixup_lr5e4",
        ],
        # large two-branch 上的正则化和 lr 消融
        "reg_large_twobranch": [
            "arch_twobranch_large_rgb_grad",
            "extra_twobranch_large_lr5e4",
            "extra_twobranch_large_lr3e4",
            "extra_twobranch_large_ls_only",
            "extra_twobranch_large_strong_ls_lr5e4",
            "extra_twobranch_large_strong_ls_mixup_lr5e4",
        ],
        # 最终 leaderboard 里关心的几个代表模型
        "leaderboard": [
            "arch_small_rgb",
            "arch_small_rgb_grad",
            "reg_single_small_rgb_grad_ls",
            "arch_large_rgb_grad",
            "extra_twobranch_large_lr3e4",
            "extra_twobranch_small_mixup_only",
        ],
    }

    for group_name, exp_list in groups.items():
        rows = []
        missing = []
        for name in exp_list:
            r = row_by_name.get(name)
            if r is None:
                missing.append(name)
            else:
                rows.append(r)

        if missing:
            print(f"[WARN] group '{group_name}' missing experiments: {', '.join(missing)}")

        out_path = os.path.join(RESULTS_DIR, f"{group_name}.csv")
        write_csv(out_path, rows)


if __name__ == "__main__":
    main()

