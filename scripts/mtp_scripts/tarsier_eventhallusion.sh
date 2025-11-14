# output_dir=${OUTPUT_DIR}
output_dir=output/tarsier_eventhallusion

vlm-benchmark --config configs/benchmark_config.yaml \
    run \
    --models "tarsier2-7b" \
    --datasets "event-hallusion" \
    --evaluators "event-hallusion" \
    --output-dir $output_dir