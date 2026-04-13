#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
```bash
python3 KlingO1-edit.py --json 版本测试集_bench_missing112.json --input_dir 原始视频 --out_dir output_missing112 >> output_missing112.log 2>&1
'''

import os
from pathlib import Path
from typing import Any, Optional, Union
import boto3

# ===================== Kling 配置 =====================
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "")
S3_BUCKET = os.getenv("S3_BUCKET", "")


# ===================== S3 配置（用环境变量控制） =====================
#S3_BUCKET = os.getenv("S3_BUCKET", "")

if not S3_BUCKET:
    pass

S3_PREFIX = os.getenv("S3_PREFIX", "kling_inputs")

PRESIGN_EXPIRE_SECONDS = int(os.getenv("PRESIGN_EXPIRE_SECONDS", "86400"))  # 24小时


# ===================== S3：上传并生成 presigned URL =====================
_s3_client: Optional[Any] = None

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def upload_and_get_url(local_path: Union[str, Path]) -> str:
    p = Path(local_path)
    if not S3_BUCKET:
        raise RuntimeError("请先设置环境变量 S3_BUCKET，例如：export S3_BUCKET='my-kling-inputs'")

    s3 = get_s3_client()
    key = f"{S3_PREFIX}/{p.name}"

    extra_args = {}
    if p.suffix.lower() == ".mp4":
        extra_args["ContentType"] = "video/mp4"
    else:
        extra_args["ContentType"] = "application/octet-stream"

    s3.upload_file(
        Filename=str(p),
        Bucket=S3_BUCKET,
        Key=key,
        ExtraArgs=extra_args
    )

    presigned_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=PRESIGN_EXPIRE_SECONDS
    )
    return presigned_url

video_url = upload_and_get_url("/home/dyvm6xra/dyvm6xrauser04/yuyang/benchmark/20260325视频版本测试_500条数据/原始视频/AIVD_202602_HM_0237.mp4")
print(video_url)
