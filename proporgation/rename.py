#!/usr/bin/env python3
"""
将 output/ 目录下的视频文件名改为与 metadata.json 中每条记录的 save_name 一致。

规则：若存在「sample_id + .mp4」，则重命名为 save_name；
若已是 save_name，则跳过；若无源文件则跳过并提示。

默认仅 dry-run，加 --apply 才执行重命名。
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="将 output 内文件重命名为 metadata 的 save_name")
    ap.add_argument(
        "--base",
        default=script_dir,
        help="数据根目录（含 metadata.json 与 output/）",
    )
    ap.add_argument(
        "--metadata",
        default="metadata.json",
        help="相对 --base 的 metadata 路径",
    )
    ap.add_argument(
        "--output-dir",
        default="output",
        help="相对 --base 的输出目录",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="执行重命名（否则只打印计划）",
    )
    args = ap.parse_args()

    meta_path = os.path.join(args.base, args.metadata)
    out_dir = os.path.join(args.base, args.output_dir)

    if not os.path.isfile(meta_path):
        print(f"找不到 metadata: {meta_path}", file=sys.stderr)
        return 1
    if not os.path.isdir(out_dir):
        print(f"找不到 output 目录: {out_dir}", file=sys.stderr)
        return 1

    with open(meta_path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        print("metadata 应为 JSON 数组", file=sys.stderr)
        return 1

    files = set(os.listdir(out_dir))
    planned: list[tuple[str, str]] = []
    skip_ok: list[str] = []
    no_src: list[tuple[str, str]] = []

    for it in items:
        sn = (it.get("save_name") or "").strip()
        sid = (it.get("sample_id") or "").strip()
        if not sn:
            continue
        if sn in files:
            skip_ok.append(sn)
            continue
        src_name = f"{sid}.mp4" if sid else ""
        if not src_name or src_name not in files:
            no_src.append((sid or "(无sample_id)", sn))
            continue
        if sn == src_name:
            skip_ok.append(sn)
            continue
        planned.append((src_name, sn))

    print(f"output 目录: {out_dir}")
    print(f"metadata 条数: {len(items)}")
    print(f"已是 save_name，跳过: {len(skip_ok)}")
    print(f"无对应源文件 (sample_id.mp4 不存在): {len(no_src)}")
    print(f"待重命名: {len(planned)}")
    if no_src and len(no_src) <= 20:
        for sid, sn in no_src:
            print(f"  [缺源] sample_id={sid!r} -> {sn!r}")
    elif no_src:
        print(f"  （前 5 条）")
        for sid, sn in no_src[:5]:
            print(f"  [缺源] sample_id={sid!r} -> {sn!r}")

    for src, dst in planned:
        print(f"  {src}  ->  {dst}")

    if not args.apply:
        print("\n[DRY-RUN] 未执行。确认无误后加 --apply")
        return 0

    errors = 0
    for src, dst in planned:
        sp = os.path.join(out_dir, src)
        dp = os.path.join(out_dir, dst)
        if os.path.lexists(dp):
            print(f"[跳过] 目标已存在: {dst}", file=sys.stderr)
            errors += 1
            continue
        try:
            os.rename(sp, dp)
            files.discard(src)
            files.add(dst)
        except OSError as e:
            print(f"[失败] {src} -> {dst}: {e}", file=sys.stderr)
            errors += 1
    print(f"\n完成，失败 {errors} 条。")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
