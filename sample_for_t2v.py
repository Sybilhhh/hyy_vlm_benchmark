#!/usr/bin/env python3
"""
从 test.csv（由 JSONL 转出来的 CSV）中做“分类分层抽样”，默认抽样比例 1:5（20%）。

核心目标：
- 抽样后样本尽可能“多样化”：优先覆盖更多 category_L4_cn（或其它主类），并在 motion / shot_type / source 等维度上均衡。
- 可选做基础过滤（is_valid / porn / terrorism / politics 等），避免低质量或不适合的样本。

输出：
- sampled_for_t2v.csv：保留原始列，并新增一列 t2v_text（优先从 prompt/caption/short_caption/long_caption 中挑一个）

用法示例：
  python3 sample_for_t2v.py \
    --input /home/dyvm6xra/dyvm6xrauser04/yuyang/models/test.csv \
    --output /home/dyvm6xra/dyvm6xrauser04/yuyang/models/sampled_for_t2v.csv \
    --ratio 1:5 \
    --seed 42

python3 /home/dyvm6xra/dyvm6xrauser04/yuyang/models/sample_for_t2v.py \
  --input /home/dyvm6xra/dyvm6xrauser04/yuyang/models/test.csv \
  --output /home/dyvm6xra/dyvm6xrauser04/yuyang/models/sampled_for_t2v.csv \
  --ratio 1:5 \
  --seed 42 \
  --require_valid \
  --max_porn 0 \
  --max_terrorism 0 \
  --max_politics 10

  python3 sample_for_t2v.py ... --min_col_nonempty_ratio 0
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_ratio(s: str) -> float:
    """
    支持：
    - "0.2"
    - "1:5"（解释为 1/5 = 0.2，常见语义：每 5 条取 1 条）
    """
    s = s.strip()
    if ":" in s:
        a, b = s.split(":", 1)
        num = float(a)
        den = float(b)
        if den == 0:
            raise ValueError("ratio denominator cannot be 0")
        return num / den
    return float(s)


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        xs = x.strip()
        if xs == "" or xs.lower() == "nan":
            return None
        try:
            return float(xs)
        except ValueError:
            return None
    return None


def to_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        xs = x.strip().lower()
        if xs in {"true", "t", "1", "yes", "y"}:
            return True
        if xs in {"false", "f", "0", "no", "n"}:
            return False
    return None


def bin_numeric(x: Optional[float], edges: Sequence[float], prefix: str) -> str:
    """
    将连续数值分桶成离散类别字符串，避免直接用 float 做分层导致桶爆炸。
    edges 需要升序，最后一个可以是 float('inf')。
    """
    if x is None:
        return ""
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if x >= lo and x < hi:
            return f"{prefix}[{lo},{hi})"
    return f"{prefix}[{edges[-1]},inf)"


def add_class_bins_inplace(row: Dict[str, Any]) -> None:
    """
    按用户指定的字段生成用于“分类/分层抽样”的离散桶字段（cls_*）。
    原字段不变；cls_* 仅用于分层 key。

    目标字段：
    - category_L1_cn, category_L2_cn, category_L3_cn, category_L4_cn
    - clip_esthetics_value, clip_quality_value, clip_vqa_value
    - duration, height, width
    - label, shot_type_en
    """
    # 直接使用的离散字段（保留原值的字符串形式）
    row["cls_category_L1_cn"] = str(row.get("category_L1_cn") or "")
    row["cls_category_L2_cn"] = str(row.get("category_L2_cn") or "")
    row["cls_category_L3_cn"] = str(row.get("category_L3_cn") or "")
    row["cls_category_L4_cn"] = str(row.get("category_L4_cn") or "")
    row["cls_label"] = str(row.get("label") or "")
    row["cls_shot_type_en"] = str(row.get("shot_type_en") or "")

    # clip 分数通常在 [0,1]，高端密一点
    prob_edges = [0, 0.7, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0000001]
    row["cls_clip_esthetics_value"] = bin_numeric(to_float(row.get("clip_esthetics_value")), prob_edges, "aes=")
    row["cls_clip_quality_value"] = bin_numeric(to_float(row.get("clip_quality_value")), prob_edges, "cq=")
    row["cls_clip_vqa_value"] = bin_numeric(to_float(row.get("clip_vqa_value")), prob_edges, "vqa=")

    # duration / height / width 分桶
    row["cls_duration"] = bin_numeric(to_float(row.get("duration")), [0, 3, 5, 10, 20, 40, 80, float("inf")], "dur=")
    row["cls_height"] = bin_numeric(to_float(row.get("height")), [0, 480, 720, 1080, 1440, 2160, 3000, float("inf")], "h=")
    row["cls_width"] = bin_numeric(to_float(row.get("width")), [0, 640, 960, 1280, 1920, 2560, 3840, 5000, float("inf")], "w=")


def pick_first_text(row: Dict[str, Any], text_fields: Sequence[str]) -> str:
    for k in text_fields:
        v = row.get(k)
        if v is None:
            continue
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def is_nonempty_cell(v: Any) -> bool:
    """
    统计“列非空占比”时使用：
    - 空串/全空白算空
    - 'NaN'（大小写不敏感）算空
    """
    if v is None:
        return False
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return False
    return True


def quality_score(row: Dict[str, Any]) -> float:
    """
    一个“足够用”的质量评分（越大越好），用于同组内优先挑更可能好用的样本。
    不追求绝对正确，可按你的偏好再调权重。
    """
    score = 0.0

    # 视频主观质量（如果有）
    mos = to_float(row.get("mos_score"))
    if mos is not None:
        score += 3.0 * mos

    real_mos = to_float(row.get("real_mos_score"))
    if real_mos is not None:
        score += 1.0 * real_mos

    # 美学/质量/问答等模型分数（如果有）
    aes = to_float(row.get("clip_esthetics_value"))
    if aes is not None:
        score += 2.0 * aes

    cq = to_float(row.get("clip_quality_value"))
    if cq is not None:
        score += 1.0 * cq

    vqa = to_float(row.get("clip_vqa_value"))
    if vqa is not None:
        score += 0.5 * vqa

    laion = to_float(row.get("laion_aesv2"))
    if laion is not None:
        score += 0.2 * laion

    # 轻微惩罚可能影响 T2V 训练的元素（有字幕/水印/Logo/黑边等）
    border = to_float(row.get("border_value")) or 0.0
    logo = to_float(row.get("logo_value")) or 0.0
    watermark = to_float(row.get("watermark_value")) or 0.0
    subtitle = to_float(row.get("subtitle_value")) or 0.0
    score -= 0.2 * border
    score -= 0.1 * logo
    score -= 0.1 * watermark
    score -= 0.05 * subtitle

    return score


def pass_filters(
    row: Dict[str, Any],
    require_valid: bool,
    max_politics: Optional[float],
    max_porn: Optional[float],
    max_terrorism: Optional[float],
) -> bool:
    if require_valid:
        is_valid = to_bool(row.get("is_valid"))
        if is_valid is False:
            return False

    if max_politics is not None:
        v = to_float(row.get("frame_politics_value"))
        if v is not None and v > max_politics:
            return False

    if max_porn is not None:
        v = to_float(row.get("frame_porn_value"))
        if v is not None and v > max_porn:
            return False

    if max_terrorism is not None:
        v = to_float(row.get("frame_terrorism_value"))
        if v is not None and v > max_terrorism:
            return False

    return True


def group_key(row: Dict[str, Any], fields: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for f in fields:
        v = row.get(f)
        if v is None:
            out.append("")
        else:
            out.append(str(v))
    return tuple(out)


def round_robin_take(buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]], n: int) -> List[Dict[str, Any]]:
    """
    在多个桶之间轮询取样，尽量均衡覆盖；桶内假定已按质量降序排好。
    """
    keys = list(buckets.keys())
    # 让小桶优先，提升覆盖度
    keys.sort(key=lambda k: len(buckets[k]))

    deques = {k: deque(buckets[k]) for k in keys}
    alive = deque([k for k in keys if deques[k]])

    picked: List[Dict[str, Any]] = []
    while alive and len(picked) < n:
        k = alive.popleft()
        if not deques[k]:
            continue
        picked.append(deques[k].popleft())
        if deques[k]:
            alive.append(k)
    return picked


def stratified_diverse_sample(
    rows: List[Dict[str, Any]],
    target_n: int,
    primary_field: str,
    secondary_fields: Sequence[str],
    seed: int,
) -> List[Dict[str, Any]]:
    """
    两层分层：
    - 第一层：primary_field（默认 category_L4_cn）尽量覆盖更多类
    - 第二层：在每个 primary 类里按 secondary_fields（如 motion/shot_type）轮询取，提升多样性
    """
    rng = random.Random(seed)

    # 按主类分组
    primary_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        k = str(r.get(primary_field) or "")
        primary_groups[k].append(r)

    # 桶内按质量排序（高优先）
    for k in list(primary_groups.keys()):
        primary_groups[k].sort(key=quality_score, reverse=True)

    # 如果主类数 >= target_n：优先选样本数更多的类（再在类内选质量最高的）
    primary_keys = list(primary_groups.keys())
    primary_keys.sort(key=lambda k: len(primary_groups[k]), reverse=True)

    picked: List[Dict[str, Any]] = []
    picked_ids: set = set()

    # Stage 1：每个主类先取 1 条，尽量覆盖
    for k in primary_keys:
        if len(picked) >= target_n:
            break
        if not primary_groups[k]:
            continue
        cand = primary_groups[k][0]
        uid = cand.get("uuid") or cand.get("id") or id(cand)
        if uid in picked_ids:
            continue
        picked.append(cand)
        picked_ids.add(uid)

    if len(picked) >= target_n:
        rng.shuffle(picked)
        return picked[:target_n]

    # Stage 2：按主类规模做“近似比例”配额，再在类内用二级轮询取
    remaining = target_n - len(picked)
    total = sum(len(v) for v in primary_groups.values())
    if total <= 0:
        return picked

    # 计算每个主类的目标配额（不含 stage1 的那 1 条）
    quotas: Dict[str, int] = {}
    for k in primary_keys:
        frac = len(primary_groups[k]) / total
        quotas[k] = int(round(frac * remaining))

    # 调整配额使其总和 == remaining
    qsum = sum(quotas.values())
    if qsum < remaining:
        # 给大类补
        i = 0
        while qsum < remaining and primary_keys:
            quotas[primary_keys[i % len(primary_keys)]] += 1
            qsum += 1
            i += 1
    elif qsum > remaining:
        # 从小类扣
        i = 0
        small = list(reversed(primary_keys))
        while qsum > remaining and small:
            k = small[i % len(small)]
            if quotas[k] > 0:
                quotas[k] -= 1
                qsum -= 1
            i += 1

    # 在每个主类内做二级分桶并轮询取
    for k in primary_keys:
        take_n = quotas.get(k, 0)
        if take_n <= 0:
            continue

        # 去掉已经在 stage1 选中的第一条
        candidates = primary_groups[k][1:] if primary_groups[k] else []
        if not candidates:
            continue

        buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
        for r in candidates:
            uid = r.get("uuid") or r.get("id") or id(r)
            if uid in picked_ids:
                continue
            sk = group_key(r, secondary_fields)
            buckets[sk].append(r)

        for sk in list(buckets.keys()):
            buckets[sk].sort(key=quality_score, reverse=True)

        taken = round_robin_take(buckets, take_n)
        for r in taken:
            uid = r.get("uuid") or r.get("id") or id(r)
            if uid in picked_ids:
                continue
            picked.append(r)
            picked_ids.add(uid)
            if len(picked) >= target_n:
                break
        if len(picked) >= target_n:
            break

    # Stage 3：如果还不够（极端情况下），随机补齐
    if len(picked) < target_n:
        pool = [r for r in rows if (r.get("uuid") or r.get("id") or id(r)) not in picked_ids]
        rng.shuffle(pool)
        for r in pool:
            picked.append(r)
            if len(picked) >= target_n:
                break

    rng.shuffle(picked)
    return picked[:target_n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="输入 CSV（例如 test.csv）")
    ap.add_argument("--output", type=str, required=True, help="输出 CSV（抽样后）")
    ap.add_argument("--ratio", type=str, default="1:5", help='抽样比例，支持 "0.2" 或 "1:5"（默认 1:5=0.2）')
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--primary", type=str, default="category_L4_cn", help="主分层字段（默认 category_L4_cn）")
    ap.add_argument(
        "--secondary",
        type=str,
        # 按用户指定字段进行分类抽样：
        # category_L1/2/3/4_cn, clip_*, duration, height, width, label, shot_type_en
        # 连续字段用 cls_* 分桶
        default="cls_category_L1_cn,cls_category_L2_cn,cls_category_L3_cn,cls_category_L4_cn,"
        "cls_clip_esthetics_value,cls_clip_quality_value,cls_clip_vqa_value,"
        "cls_duration,cls_height,cls_width,cls_label,cls_shot_type_en",
        help="二级分层字段，用逗号分隔（默认按 category_L1/2/3/4_cn + clip_* + duration + height + width + label + shot_type_en 分层；连续字段会先分桶为 cls_*）",
    )

    ap.add_argument("--require_valid", action="store_true", help="若指定，则只保留 is_valid=True 的样本")
    ap.add_argument("--max_porn", type=float, default=None, help="frame_porn_value 最大允许值（例如 0）")
    ap.add_argument("--max_terrorism", type=float, default=None, help="frame_terrorism_value 最大允许值（例如 0）")
    ap.add_argument("--max_politics", type=float, default=None, help="frame_politics_value 最大允许值（例如 10）")

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
        help="只保留在抽样结果里非空占比 >= 该阈值的列（默认 0.8）。设置为 0 可关闭该过滤。",
    )

    args = ap.parse_args()

    ratio = parse_ratio(args.ratio)
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"ratio must be within (0,1], got {ratio}")

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 读 CSV
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]

    # 过滤
    filtered = [
        r
        for r in rows
        if pass_filters(
            r,
            require_valid=args.require_valid,
            max_politics=args.max_politics,
            max_porn=args.max_porn,
            max_terrorism=args.max_terrorism,
        )
    ]

    # 生成用于分类/分层抽样的 cls_* 桶字段
    for r in filtered:
        add_class_bins_inplace(r)

    target_n = max(1, int(math.floor(len(filtered) * ratio)))

    secondary_fields = [s.strip() for s in args.secondary.split(",") if s.strip()]
    text_fields = [s.strip() for s in args.text_fields.split(",") if s.strip()]

    sampled = stratified_diverse_sample(
        filtered,
        target_n=target_n,
        primary_field=args.primary,
        secondary_fields=secondary_fields,
        seed=args.seed,
    )

    # 输出时加一列 t2v_text
    out_fields = list(fieldnames)
    if "t2v_text" not in out_fields:
        out_fields.append("t2v_text")

    # 按“非空占比”过滤列（在抽样结果上统计）
    if args.min_col_nonempty_ratio and args.min_col_nonempty_ratio > 0:
        total = len(sampled)
        if total > 0:
            nonempty_counts = {c: 0 for c in out_fields}
            for r in sampled:
                # 注意：t2v_text 还没写回 r，这里用即时生成
                for c in out_fields:
                    if c == "t2v_text":
                        v = pick_first_text(r, text_fields)
                    else:
                        v = r.get(c)
                    if is_nonempty_cell(v):
                        nonempty_counts[c] += 1

            keep_fields = []
            for c in out_fields:
                if c == "t2v_text":
                    keep_fields.append(c)
                    continue
                ratio = nonempty_counts[c] / total
                if ratio >= args.min_col_nonempty_ratio:
                    keep_fields.append(c)
            out_fields = keep_fields

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for r in sampled:
            rr = dict(r)
            rr["t2v_text"] = pick_first_text(rr, text_fields)
            writer.writerow(rr)

    print(f"input_rows={len(rows)} filtered_rows={len(filtered)} ratio={ratio} sampled_rows={len(sampled)}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()


