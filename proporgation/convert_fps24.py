#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 原始视频/ 目录下所有 mp4 转换为 24 fps，保持时长不变。
用法:
  转换+提取一起跑（推荐）:
    python3 convert_fps24.py run --json 版本测试集_bench.json \
        --input_dir 原始视频 --out_dir 原始视频_24fps --keyframe_out 关键帧_24fps --workers 8

  仅帧率转换:
    python3 convert_fps24.py convert --input_dir 原始视频 --out_dir 原始视频_24fps --workers 8

  仅关键帧提取:
    python3 convert_fps24.py extract --json 版本测试集_bench.json \
        --orig_dir 原始视频 --video_dir 原始视频_24fps --keyframe_out 关键帧_24fps --workers 8
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

TARGET_FPS = 24
FPS_TOLERANCE = 0.02  # ±0.02 fps 以内视为已达标

FFMPEG = shutil.which("ffmpeg") or "/scratch/dyvm6xra/dyvm6xrauser04/miniforge3/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/scratch/dyvm6xra/dyvm6xrauser04/miniforge3/bin/ffprobe"


def probe(video: Path) -> dict:
    cmd = [
        FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,duration,codec_name",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    info = json.loads(out)
    stream = info["streams"][0]
    num, den = map(int, stream["r_frame_rate"].split("/"))
    fps = num / den
    duration = float(stream.get("duration") or info["format"]["duration"])
    return {"fps": fps, "duration": duration, "codec": stream.get("codec_name", "")}


def convert_one(video: Path, out_dir: Path | None) -> str:
    try:
        meta = probe(video)
    except Exception as e:
        return f"[SKIP]  {video.name}  probe 失败: {e}"

    dst = (out_dir / video.name) if out_dir else video
    in_place = (out_dir is None)

    if abs(meta["fps"] - TARGET_FPS) <= FPS_TOLERANCE:
        if not in_place and not dst.exists():
            shutil.copy2(str(video), str(dst))
        return f"[OK]    {video.name}  已是 {meta['fps']:.2f} fps，跳过"

    write_dir = dst.parent
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=write_dir)
    tmp = Path(tmp_path)
    try:
        cmd = [
            FFMPEG, "-y", "-i", str(video),
            "-vf", f"fps={TARGET_FPS}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(tmp),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        new_meta = probe(tmp)
        dur_diff = abs(new_meta["duration"] - meta["duration"])
        if dur_diff > 0.15:
            tmp.unlink(missing_ok=True)
            return (f"[WARN]  {video.name}  时长偏差 {dur_diff:.3f}s "
                    f"(原 {meta['duration']:.3f}s → {new_meta['duration']:.3f}s)，未写入")

        shutil.move(str(tmp), str(dst))
        return (f"[DONE]  {video.name}  {meta['fps']:.2f} → {new_meta['fps']:.2f} fps  "
                f"时长 {meta['duration']:.3f}s → {new_meta['duration']:.3f}s")
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        return f"[ERR]   {video.name}  ffmpeg 失败: {e.stderr[-300:] if e.stderr else e}"
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return f"[ERR]   {video.name}  {e}"
    finally:
        import os
        try:
            os.close(tmp_fd)
        except OSError:
            pass


def extract_one(item: dict, orig_dir: Path, video_dir: Path, keyframe_out: Path) -> str:
    cid = item.get("cid", "?")
    video_name = item.get("原始视频")
    kf_name = item.get("关键帧")
    idx = item.get("idx")

    if not video_name or idx is None or not kf_name:
        return f"[SKIP]  {cid}  缺少 原始视频/idx/关键帧 字段"

    dst = keyframe_out / kf_name
    if dst.exists():
        return f"[OK]    {cid}  {kf_name} 已存在，跳过"

    orig_video = orig_dir / video_name
    fps24_video = video_dir / video_name

    if not fps24_video.exists():
        return f"[ERR]   {cid}  24fps 视频不存在: {fps24_video.name}"

    try:
        orig_meta = probe(orig_video) if orig_video.exists() else None
    except Exception:
        orig_meta = None

    if orig_meta:
        orig_fps = orig_meta["fps"]
    else:
        orig_fps = TARGET_FPS

    timestamp = idx / orig_fps
    nearest_frame = round(timestamp * TARGET_FPS)

    try:
        cmd = [
            FFMPEG, "-y", "-i", str(fps24_video),
            "-vf", f"select=eq(n\\,{nearest_frame})",
            "-vsync", "vfr",
            "-frames:v", "1",
            "-q:v", "1",
            str(dst),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        if not dst.exists() or dst.stat().st_size == 0:
            dst.unlink(missing_ok=True)
            return f"[ERR]   {cid}  提取帧为空 (frame={nearest_frame}, t={timestamp:.4f}s)"

        return (f"[DONE]  {cid}  orig_fps={orig_fps:.2f} idx={idx} → "
                f"t={timestamp:.4f}s → 24fps_frame={nearest_frame}  → {kf_name}")
    except subprocess.CalledProcessError as e:
        dst.unlink(missing_ok=True)
        stderr = e.stderr.decode(errors="replace")[-300:] if e.stderr else str(e)
        return f"[ERR]   {cid}  ffmpeg 失败: {stderr}"
    except Exception as e:
        dst.unlink(missing_ok=True)
        return f"[ERR]   {cid}  {e}"


# ===================== 转换 + 提取合并 =====================
def convert_and_extract(video: Path, out_dir: Path, items: list, keyframe_out: Path) -> list[str]:
    """先转换该视频到 24fps，再立即提取该视频的所有关键帧，返回每步的消息列表。"""
    msgs = []

    # --- 转换 ---
    try:
        meta = probe(video)
    except Exception as e:
        msg = f"[SKIP]  {video.name}  probe 失败: {e}"
        msgs.append(msg)
        return msgs

    dst = out_dir / video.name
    orig_fps = meta["fps"]

    if abs(orig_fps - TARGET_FPS) <= FPS_TOLERANCE:
        if not dst.exists():
            shutil.copy2(str(video), str(dst))
        msgs.append(f"[OK]    {video.name}  已是 {orig_fps:.2f} fps，复制到输出目录")
    else:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=out_dir)
        tmp = Path(tmp_path)
        try:
            cmd = [
                FFMPEG, "-y", "-i", str(video),
                "-vf", f"fps={TARGET_FPS}",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                "-movflags", "+faststart",
                str(tmp),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            new_meta = probe(tmp)
            dur_diff = abs(new_meta["duration"] - meta["duration"])
            if dur_diff > 0.15:
                tmp.unlink(missing_ok=True)
                msgs.append(f"[WARN]  {video.name}  时长偏差 {dur_diff:.3f}s，跳过关键帧提取")
                return msgs
            shutil.move(str(tmp), str(dst))
            msgs.append(f"[DONE]  {video.name}  {orig_fps:.2f} → {TARGET_FPS} fps  "
                        f"时长 {meta['duration']:.3f}s → {new_meta['duration']:.3f}s")
        except subprocess.CalledProcessError as e:
            tmp.unlink(missing_ok=True)
            stderr = e.stderr.decode(errors="replace")[-200:] if e.stderr else str(e)
            msgs.append(f"[ERR]   {video.name}  转换失败: {stderr}")
            return msgs
        except Exception as e:
            tmp.unlink(missing_ok=True)
            msgs.append(f"[ERR]   {video.name}  {e}")
            return msgs
        finally:
            import os
            try:
                os.close(tmp_fd)
            except OSError:
                pass

    # --- 提取关键帧 ---
    for item in items:
        cid = item.get("cid", "?")
        kf_name = item.get("关键帧")
        idx = item.get("idx")
        if idx is None or not kf_name:
            msgs.append(f"  [SKIP]  {cid}  缺少 idx/关键帧 字段")
            continue

        kf_dst = keyframe_out / kf_name
        if kf_dst.exists():
            msgs.append(f"  [OK]    {cid}  {kf_name} 已存在，跳过")
            continue

        timestamp = idx / orig_fps
        nearest_frame = round(timestamp * TARGET_FPS)
        try:
            cmd = [
                FFMPEG, "-y", "-i", str(dst),
                "-vf", f"select=eq(n\\,{nearest_frame})",
                "-vsync", "vfr",
                "-frames:v", "1",
                "-q:v", "1",
                str(kf_dst),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            if not kf_dst.exists() or kf_dst.stat().st_size == 0:
                kf_dst.unlink(missing_ok=True)
                msgs.append(f"  [ERR]   {cid}  提取帧为空 (frame={nearest_frame})")
            else:
                msgs.append(f"  [FRAME] {cid}  idx={idx} orig_fps={orig_fps:.2f} "
                            f"→ t={timestamp:.4f}s → frame={nearest_frame} → {kf_name}")
        except subprocess.CalledProcessError as e:
            kf_dst.unlink(missing_ok=True)
            stderr = e.stderr.decode(errors="replace")[-200:] if e.stderr else str(e)
            msgs.append(f"  [ERR]   {cid}  提取失败: {stderr}")
        except Exception as e:
            kf_dst.unlink(missing_ok=True)
            msgs.append(f"  [ERR]   {cid}  {e}")

    return msgs


def cmd_run(args):
    base = Path(__file__).resolve().parent
    json_path = (base / args.json).resolve()
    input_dir = (base / args.input_dir).resolve()
    out_dir = (base / args.out_dir).resolve()
    keyframe_out = (base / args.keyframe_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    keyframe_out.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # 按视频名分组
    from collections import defaultdict
    video_to_items: dict[str, list] = defaultdict(list)
    for item in data:
        vname = item.get("原始视频")
        if vname:
            video_to_items[vname].append(item)

    # 找出输入目录中实际存在的视频（也处理 JSON 未提及的视频）
    all_videos = sorted(input_dir.glob("*.mp4"))
    tasks = []
    for v in all_videos:
        items_for_v = video_to_items.get(v.name, [])
        tasks.append((v, items_for_v))

    print(f"找到 {len(all_videos)} 个视频，{len(data)} 条 JSON 记录，并发 {args.workers}")
    print(f"24fps 输出: {out_dir}\n关键帧输出: {keyframe_out}")

    conv_done = conv_skip = conv_err = kf_done = kf_skip = kf_err = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(convert_and_extract, v, out_dir, items, keyframe_out): v
            for v, items in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            msgs = fut.result()
            for msg in msgs:
                print(f"[{i}/{len(tasks)}] {msg}")
                stripped = msg.strip()
                if stripped.startswith("[DONE]"):
                    conv_done += 1
                elif stripped.startswith("[OK]") or stripped.startswith("[SKIP]"):
                    conv_skip += 1
                elif stripped.startswith("[WARN]") or stripped.startswith("[ERR]"):
                    conv_err += 1
                elif stripped.startswith("[FRAME]"):
                    kf_done += 1

    print(f"\n转换: 完成 {conv_done}，跳过 {conv_skip}，失败/警告 {conv_err}")
    print(f"关键帧: 提取 {kf_done}，跳过 {kf_skip}，失败 {kf_err}")


# ===================== CLI =====================
def cmd_convert(args):
    base = Path(__file__).resolve().parent
    input_dir = (base / args.input_dir).resolve()

    out_dir = None
    if args.out_dir:
        out_dir = (base / args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {out_dir}")
    else:
        print("未指定 --out_dir，将原地替换")

    videos = sorted(input_dir.glob("*.mp4"))
    print(f"找到 {len(videos)} 个 mp4，目标 {TARGET_FPS} fps，并发 {args.workers}")

    done = skip = warn = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_one, v, out_dir): v for v in videos}
        for i, fut in enumerate(as_completed(futures), 1):
            msg = fut.result()
            print(f"[{i}/{len(videos)}] {msg}")
            if msg.startswith("[DONE]"):
                done += 1
            elif msg.startswith("[OK]") or msg.startswith("[SKIP]"):
                skip += 1
            elif msg.startswith("[WARN]"):
                warn += 1
            else:
                err += 1

    print(f"\n完成: 转换 {done}，跳过 {skip}，警告 {warn}，失败 {err}")


def cmd_extract(args):
    base = Path(__file__).resolve().parent
    json_path = (base / args.json).resolve()
    orig_dir = (base / args.orig_dir).resolve()
    video_dir = (base / args.video_dir).resolve()
    keyframe_out = (base / args.keyframe_out).resolve()
    keyframe_out.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"读取 {len(data)} 条记录，原始视频目录: {orig_dir}")
    print(f"24fps 视频目录: {video_dir}，关键帧输出: {keyframe_out}")

    done = skip = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_one, item, orig_dir, video_dir, keyframe_out): item
            for item in data
        }
        for i, fut in enumerate(as_completed(futures), 1):
            msg = fut.result()
            print(f"[{i}/{len(data)}] {msg}")
            if msg.startswith("[DONE]"):
                done += 1
            elif msg.startswith("[OK]") or msg.startswith("[SKIP]"):
                skip += 1
            else:
                err += 1

    print(f"\n完成: 提取 {done}，跳过 {skip}，失败 {err}")


def main():
    parser = argparse.ArgumentParser(description="视频帧率转换 & 关键帧提取")
    sub = parser.add_subparsers(dest="command")

    p_conv = sub.add_parser("convert", help="批量转换视频帧率到 24 fps")
    p_conv.add_argument("--input_dir", type=str, default="原始视频")
    p_conv.add_argument("--out_dir", type=str, default=None,
                        help="输出目录（不指定则原地替换）")
    p_conv.add_argument("--workers", type=int, default=8)

    p_ext = sub.add_parser("extract", help="从 24fps 视频提取最邻近关键帧")
    p_ext.add_argument("--json", type=str, required=True, help="bench JSON 文件")
    p_ext.add_argument("--orig_dir", type=str, default="原始视频",
                       help="原始视频目录（用于读取原始帧率）")
    p_ext.add_argument("--video_dir", type=str, default="原始视频_24fps",
                       help="24fps 视频目录")
    p_ext.add_argument("--keyframe_out", type=str, default="关键帧_24fps",
                       help="关键帧输出目录")
    p_ext.add_argument("--workers", type=int, default=8)

    p_run = sub.add_parser("run", help="转换帧率并立即提取关键帧（推荐）")
    p_run.add_argument("--json", type=str, required=True, help="bench JSON 文件")
    p_run.add_argument("--input_dir", type=str, default="原始视频")
    p_run.add_argument("--out_dir", type=str, default="原始视频_24fps",
                       help="24fps 视频输出目录")
    p_run.add_argument("--keyframe_out", type=str, default="关键帧_24fps",
                       help="关键帧输出目录")
    p_run.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "extract":
        cmd_extract(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
