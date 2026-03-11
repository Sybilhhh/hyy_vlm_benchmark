K5_DEBUG_PROP_KEYFRAME=1 python infer_unified.py \
  --conf_path configs/k5_unified.yaml \
  --dit_checkpoint <你的ckpt路径> \
  --csv_path <含有prop或keyframe任务的CSV> \
  --output_dir <输出目录> \
  --task prop   # 或 --task keyframe，或者CSV里自己有task列
