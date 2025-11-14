#!/bin/bash

# 便捷脚本：对比两个模型的输出
# 使用方法: ./scripts/compare_models.sh model_a model_b dataset [output_dir]

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 帮助信息
show_help() {
    echo -e "${BLUE}=== VLM Benchmark Model Comparison Script ===${NC}"
    echo ""
    echo "Usage: $0 <model_a> <model_b> <dataset> [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  model_a     : Name of the first model (e.g., qwen3vl-8b)"
    echo "  model_b     : Name of the second model (e.g., qwen2.5vl-7b)"
    echo "  dataset     : Dataset to use (e.g., test-50)"
    echo "  output_dir  : Output directory (default: ./output/comparison)"
    echo ""
    echo "Example:"
    echo "  $0 qwen3vl-8b qwen2.5vl-7b test-50"
    echo "  $0 qwen2.5vl-32b qwen2.5vl-7b test-50 ./output/my_comparison"
    echo ""
    exit 0
}

# 检查参数
if [ "$#" -lt 3 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    show_help
fi

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
fi

MODEL_A=$1
MODEL_B=$2
DATASET=$3
OUTPUT_DIR=${4:-"./output/comparison_${MODEL_A}_vs_${MODEL_B}"}

BENCHMARK_DIR="/home/dyvm6xra/dyvm6xrauser04/yuyang/vlm_benchmark"
cd $BENCHMARK_DIR

echo -e "${BLUE}=== Model Comparison Pipeline ===${NC}"
echo -e "Model A: ${GREEN}${MODEL_A}${NC}"
echo -e "Model B: ${GREEN}${MODEL_B}${NC}"
echo -e "Dataset: ${GREEN}${DATASET}${NC}"
echo -e "Output:  ${GREEN}${OUTPUT_DIR}${NC}"
echo ""

# 步骤 1: 运行模型 A
echo -e "${YELLOW}[Step 1/3] Running Model A: ${MODEL_A}${NC}"
MODEL_A_OUTPUT="${OUTPUT_DIR}/predictions_${MODEL_A}"

if [ -f "${MODEL_A_OUTPUT}/predictions/predictions.json" ]; then
    echo -e "${GREEN}✓ Predictions for ${MODEL_A} already exist, skipping...${NC}"
else
    vlm-benchmark run -m ${MODEL_A} -d ${DATASET} -o ${MODEL_A_OUTPUT}
    echo -e "${GREEN}✓ Model A predictions completed${NC}"
fi
echo ""

# 步骤 2: 运行模型 B
echo -e "${YELLOW}[Step 2/3] Running Model B: ${MODEL_B}${NC}"
MODEL_B_OUTPUT="${OUTPUT_DIR}/predictions_${MODEL_B}"

if [ -f "${MODEL_B_OUTPUT}/predictions/predictions.json" ]; then
    echo -e "${GREEN}✓ Predictions for ${MODEL_B} already exist, skipping...${NC}"
else
    vlm-benchmark run -m ${MODEL_B} -d ${DATASET} -o ${MODEL_B_OUTPUT}
    echo -e "${GREEN}✓ Model B predictions completed${NC}"
fi
echo ""

# 步骤 3: 创建临时评估配置
echo -e "${YELLOW}[Step 3/3] Running Pairwise Comparison${NC}"
TEMP_CONFIG="${OUTPUT_DIR}/temp_evaluator_config.yaml"

cat > ${TEMP_CONFIG} << EOF
evaluators:
  temp-comparison:
    type: pairwise-ranking
    model_a_predictions: "${MODEL_A_OUTPUT}/predictions/predictions.json"
    model_b_predictions: "${MODEL_B_OUTPUT}/predictions/predictions.json"
    model_a_name: "${MODEL_A}"
    model_b_name: "${MODEL_B}"
    judge_model: "gpt-4o"
    evaluation_criteria:
      - "Accuracy: How accurate is the description?"
      - "Detail and Completeness: Does it cover all important aspects?"
      - "Technical Terminology: Correct use of terminology?"
      - "Clarity and Coherence: Is it well-organized and clear?"
    model_name: "gpt-4o"
    endpoint: "https://openaieastus2instance.openai.azure.com/openai/deployments/gpt-4o-cv-chx0812/chat/completions?api-version=2025-01-01-preview"
    api_version: "2024-12-01-preview"
    api_key: "990a353da7b44bef8466402378c486cd"
    max_tokens: 1024
    temperature: 0
    top_p: 1.0
EOF

# 注意: 这里需要实现一个支持临时配置的方式
# 暂时使用 Python 脚本直接调用
python3 << PYTHON_SCRIPT
import asyncio
import sys
sys.path.insert(0, '${BENCHMARK_DIR}')

from evaluators import PairwiseRankingEvaluator
import json
from pathlib import Path

config = {
    'model_a_predictions': '${MODEL_A_OUTPUT}/predictions/predictions.json',
    'model_b_predictions': '${MODEL_B_OUTPUT}/predictions/predictions.json',
    'model_a_name': '${MODEL_A}',
    'model_b_name': '${MODEL_B}',
    'judge_model': 'gpt-4o',
    'evaluation_criteria': [
        'Accuracy: How accurate is the description?',
        'Detail and Completeness: Does it cover all important aspects?',
        'Technical Terminology: Correct use of terminology?',
        'Clarity and Coherence: Is it well-organized and clear?'
    ],
    'model_name': 'gpt-4o',
    'endpoint': 'https://openaieastus2instance.openai.azure.com/openai/deployments/gpt-4o-cv-chx0812/chat/completions?api-version=2025-01-01-preview',
    'api_version': '2024-12-01-preview',
    'api_key': '990a353da7b44bef8466402378c486cd',
    'max_tokens': 1024,
    'temperature': 0,
    'top_p': 1.0
}

async def main():
    evaluator = PairwiseRankingEvaluator(config)
    result = await evaluator.evaluate([], None)
    
    # 保存结果
    output_dir = Path('${OUTPUT_DIR}/evaluation_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump({
            'overall_score': result.overall_score,
            'detailed_scores': result.detailed_scores,
            'metadata': result.metadata
        }, f, indent=2)
    
    print(f'\\n✓ Results saved to: {output_dir}/summary.json')

asyncio.run(main())
PYTHON_SCRIPT

echo -e "${GREEN}✓ Pairwise comparison completed${NC}"
echo ""

# 显示结果摘要
echo -e "${BLUE}=== Comparison Results ===${NC}"
cat "${OUTPUT_DIR}/evaluation_results/summary.json" | python3 -m json.tool | head -20
echo ""
echo -e "${GREEN}Full results saved to: ${OUTPUT_DIR}/evaluation_results/${NC}"
echo -e "${BLUE}=== Done! ===${NC}"

