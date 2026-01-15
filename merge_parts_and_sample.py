#!/usr/bin/env python3
"""
遍历 /home/dyvm6xra/dyvm6xrauser04/yuyang/models/part1 到 part2 里的所有 .json，
合并写入一个总 CSV，并按 1:5 做分类分层抽样（抽样逻辑复用 sample_for_t2v.py）。

默认行为：
- 输入支持 JSON Lines（每行一个对象）或 JSON array（一个文件是一个数组）
- 默认去掉 video_embedding（太大）
- 总输出：all_merged.csv
- 抽样输出：sampled_1_5.csv（并生成 t2v_text；同时可按列非空占比过滤，默认 0.8，与 sample_for_t2v.py 一致）

用法示例：
  python /home/dyvm6xra/dyvm6xrauser04/yuyang/models/merge_parts_and_sample.py \
    --input_dirs /home/dyvm6xra/dyvm6xrauser04/yuyang/models/part1,/home/dyvm6xra/dyvm6xrauser04/yuyang/models/part2 \
    --all_csv /home/dyvm6xra/dyvm6xrauser04/yuyang/models/all_parts_merged.csv \
    --sample_csv /home/dyvm6xra/dyvm6xrauser04/yuyang/models/all_parts_sampled_1_5.csv \
    --ratio 1:5 --seed 42 --drop_video_embedding
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def flatten_value(value: Any) -> Any:
    """把 list/dict 变成 JSON 字符串，便于写 CSV；其它类型原样返回。"""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def iter_json_objects(file_path: Path) -> Iterable[Dict[str, Any]]:
    """
    支持：
    - JSONL：每行一个 JSON object
    - JSON array：文件是一个数组，数组元素为 object
    """
    # 先读一点点判断格式
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        head = f.read(2048)

    head_strip = head.lstrip()
    if not head_strip:
        return  # empty

    # JSON array
    if head_strip.startswith("["):
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        return

    # JSONL（fallback）
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def find_json_files(input_dirs: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for d in input_dirs:
        if not d.exists():
            continue
        for root, _, fnames in os.walk(d):
            for name in fnames:
                if name.lower().endswith(".json"):
                    files.append(Path(root) / name)
    files.sort()
    return files


def is_nonempty_cell(v: Any) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dirs",
        type=str,
        default="/home/dyvm6xra/dyvm6xrauser04/yuyang/models/part1,/home/dyvm6xra/dyvm6xrauser04/yuyang/models/part2",
        help="逗号分隔的输入目录（会递归扫描 .json）",
    )
    ap.add_argument(
        "--all_csv",
        type=str,
        default="/home/dyvm6xra/dyvm6xrauser04/yuyang/models/all_parts_merged.csv",
        help="合并后的总 CSV 输出路径",
    )
    ap.add_argument(
        "--sample_csv",
        type=str,
        default="/home/dyvm6xra/dyvm6xrauser04/yuyang/models/all_parts_sampled_1_5.csv",
        help="抽样后的 CSV 输出路径",
    )
    ap.add_argument("--ratio", type=str, default="1:5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop_video_embedding", action="store_true", help="丢弃 video_embedding 列（建议开启）")
    ap.add_argument("--add_source_file", action="store_true", help="添加 source_file 列用于追溯来源文件路径")

    # 抽样过滤参数：保持与 sample_for_t2v.py 一致
    ap.add_argument("--require_valid", action="store_true")
    ap.add_argument("--max_porn", type=float, default=None)
    ap.add_argument("--max_terrorism", type=float, default=None)
    ap.add_argument("--max_politics", type=float, default=None)
    ap.add_argument(
        "--text_fields",
        type=str,
        default="prompt,long_prompt,caption,short_caption,long_caption,original_caption",
        help="用于生成 t2v_text 的字段优先级（逗号分隔）",
    )
    ap.add_argument(
        "--min_col_nonempty_ratio",
        type=float,
        default=0.8,
        help="抽样输出只保留非空占比 >= 该阈值的列（默认 0.8）；设为 0 关闭",
    )
    args = ap.parse_args()

    input_dirs = [Path(p.strip()) for p in args.input_dirs.split(",") if p.strip()]
    json_files = find_json_files(input_dirs)

    # 读取并合并
    rows: List[Dict[str, Any]] = []
    all_cols: set[str] = set()
    for fp in json_files:
        for obj in iter_json_objects(fp):
            if args.drop_video_embedding:
                obj.pop("video_embedding", None)
            if args.add_source_file:
                obj["source_file"] = str(fp)

            # flatten
            flat = {k: flatten_value(v) for k, v in obj.items()}
            rows.append(flat)
            all_cols.update(flat.keys())

    all_cols_sorted = sorted(all_cols)

    # 写总 CSV
    all_csv_path = Path(args.all_csv)
    all_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with all_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_cols_sorted, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # --- 抽样（复用 sample_for_t2v.py 的逻辑）---
    # 这里直接 import 同目录下的 sample_for_t2v.py
    import importlib
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    s = importlib.import_module("sample_for_t2v")

    ratio = s.parse_ratio(args.ratio)
    filtered = [
        r
        for r in rows
        if s.pass_filters(
            r,
            require_valid=args.require_valid,
            max_politics=args.max_politics,
            max_porn=args.max_porn,
            max_terrorism=args.max_terrorism,
        )
    ]

    for r in filtered:
        s.add_class_bins_inplace(r)

    target_n = max(1, int(math.floor(len(filtered) * ratio)))
    secondary_fields = [x.strip() for x in str(getattr(s, "DEFAULT_SECONDARY", "")).split(",") if x.strip()]
    # 如果 sample_for_t2v.py 没有 DEFAULT_SECONDARY，就沿用它 argparse 里默认 secondary 字符串
    if not secondary_fields:
        secondary_fields = [
            "cls_category_L1_cn",
            "cls_category_L2_cn",
            "cls_category_L3_cn",
            "cls_category_L4_cn",
            "cls_clip_esthetics_value",
            "cls_clip_quality_value",
            "cls_clip_vqa_value",
            "cls_duration",
            "cls_height",
            "cls_width",
            "cls_label",
            "cls_shot_type_en",
        ]

    sampled = s.stratified_diverse_sample(
        filtered,
        target_n=target_n,
        primary_field="category_L4_cn",
        secondary_fields=secondary_fields,
        seed=args.seed,
    )

    text_fields = [x.strip() for x in args.text_fields.split(",") if x.strip()]

    # 抽样输出列：基于总列 + t2v_text，并做非空占比过滤
    out_fields = list(all_cols_sorted)
    if "t2v_text" not in out_fields:
        out_fields.append("t2v_text")

    if args.min_col_nonempty_ratio and args.min_col_nonempty_ratio > 0 and len(sampled) > 0:
        nonempty_counts = {c: 0 for c in out_fields}
        total = len(sampled)
        for r in sampled:
            for c in out_fields:
                v = s.pick_first_text(r, text_fields) if c == "t2v_text" else r.get(c)
                if is_nonempty_cell(v):
                    nonempty_counts[c] += 1

        keep = []
        for c in out_fields:
            if c == "t2v_text":
                keep.append(c)
                continue
            if nonempty_counts[c] / total >= args.min_col_nonempty_ratio:
                keep.append(c)
        out_fields = keep

    sample_csv_path = Path(args.sample_csv)
    sample_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for r in sampled:
            rr = dict(r)
            rr["t2v_text"] = s.pick_first_text(rr, text_fields)
            w.writerow(rr)

    print(f"json_files={len(json_files)} total_rows={len(rows)} filtered_rows={len(filtered)} sampled_rows={len(sampled)}")
    print(f"all_csv={all_csv_path}")
    print(f"sample_csv={sample_csv_path}")


if __name__ == "__main__":
    main()


