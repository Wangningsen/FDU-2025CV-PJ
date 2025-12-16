#!/usr/bin/env bash
set -e

ROOT="/data3/nwang60/dataset/aigcdetectbenchmark"

cd "$ROOT"

for f in *.tar.gz; do
  # 跳过不是文件的情况
  [ -f "$f" ] || continue

  base="${f%.tar.gz}"  # 去掉 .tar.gz
  echo "== 解压 $f -> $base/ =="

  mkdir -p "$base"
  # 解压到对应目录下
  tar -xzf "$f" -C "$base"
done

echo "全部解压完成。现在结构应该是:"
echo "  $ROOT/<subset_name>/{0_real,1_fake,...}"
