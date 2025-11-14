# output_dir=${OUTPUT_DIR}
# pip install -e .

output_dir=output/test3

vlm-benchmark --config configs/benchmark_config.yaml \
    run \
    --models "tarsier2-7b" \
    --datasets "dream-1k" \
    --evaluators "autodq" \
    --output-dir $output_dir

# output_dir=$XXX/test
# echo $output_dir