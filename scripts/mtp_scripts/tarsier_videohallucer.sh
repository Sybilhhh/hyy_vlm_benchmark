# output_dir=${OUTPUT_DIR}
output_dir=output/tarsier_videohallucer

vlm-benchmark --config configs/benchmark_config.yaml \
    run \
    --models "tarsier2-7b" \
    --datasets "video-hallucer" \
    --evaluators "video-hallucer" \
    --output-dir $output_dir