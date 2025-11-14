#!/usr/bin/env python3
"""
便捷脚本：对比两个模型的输出

使用方法:
    python scripts/compare_models.py qwen3vl-8b qwen2.5vl-7b \\
        --pred-a ./output/qwen3vl_8b/predictions.json \\
        --pred-b ./output/qwen25vl_7b/predictions.json \\
        --output ./output/comparison
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluators.pairwise_ranking import PairwiseRankingEvaluator
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare two VLM models using pairwise ranking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python scripts/compare_models.py qwen3vl-8b qwen2.5vl-7b \\
      --pred-a ./output/qwen3vl_8b/test50/predictions/predictions.json \\
      --pred-b ./output/qwen25vl_7b/test50/predictions/predictions.json
  
  # With custom output directory
  python scripts/compare_models.py qwen3vl-8b qwen2.5vl-7b \\
      --pred-a ./output/qwen3vl_8b/test50/predictions/predictions.json \\
      --pred-b ./output/qwen25vl_7b/test50/predictions/predictions.json \\
      --output ./output/my_comparison
  
  # With custom criteria
  python scripts/compare_models.py qwen3vl-8b qwen2.5vl-7b \\
      --pred-a ./output/qwen3vl_8b/test50/predictions/predictions.json \\
      --pred-b ./output/qwen25vl_7b/test50/predictions/predictions.json \\
      --criteria "Accuracy" "Completeness" "Clarity"
        """
    )
    
    # 必需参数
    parser.add_argument('model_a', type=str, help='Name of model A (for display)')
    parser.add_argument('model_b', type=str, help='Name of model B (for display)')
    
    # 预测文件路径
    parser.add_argument('--pred-a', '--predictions-a', required=True,
                       help='Path to model A predictions JSON file')
    parser.add_argument('--pred-b', '--predictions-b', required=True,
                       help='Path to model B predictions JSON file')
    
    # 可选参数
    parser.add_argument('--output', '-o', default='./output/comparison',
                       help='Output directory (default: ./output/comparison)')
    parser.add_argument('--judge', default='gpt-4o',
                       help='Judge model name (default: gpt-4o)')
    parser.add_argument('--criteria', nargs='+',
                       help='Evaluation criteria (space-separated)')
    
    # GPT-4o API 配置
    parser.add_argument('--endpoint',
                       default='https://openaieastus2instance.openai.azure.com/openai/deployments/gpt-4o-cv-chx0812/chat/completions?api-version=2025-01-01-preview',
                       help='Azure OpenAI endpoint')
    parser.add_argument('--api-key',
                       default='990a353da7b44bef8466402378c486cd',
                       help='Azure OpenAI API key')
    parser.add_argument('--api-version', default='2024-12-01-preview',
                       help='Azure OpenAI API version')
    
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # 验证输入文件
    pred_a_path = Path(args.pred_a)
    pred_b_path = Path(args.pred_b)
    
    if not pred_a_path.exists():
        logger.error(f"Model A predictions not found: {pred_a_path}")
        sys.exit(1)
    
    if not pred_b_path.exists():
        logger.error(f"Model B predictions not found: {pred_b_path}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建配置
    config = {
        'model_a_predictions': str(pred_a_path.absolute()),
        'model_b_predictions': str(pred_b_path.absolute()),
        'model_a_name': args.model_a,
        'model_b_name': args.model_b,
        'judge_model': args.judge,
        'model_name': args.judge,
        'endpoint': args.endpoint,
        'api_key': args.api_key,
        'api_version': args.api_version,
        'max_tokens': 1024,
        'temperature': 0,
        'top_p': 1.0
    }
    
    # 添加评估标准（如果指定）
    if args.criteria:
        config['evaluation_criteria'] = args.criteria
    else:
        # 默认标准（针对 Test-50 电影制作场景）
        config['evaluation_criteria'] = [
            'Accuracy: How accurate is the description of camera and lighting setup?',
            'Detail and Completeness: Does the response cover all important aspects (camera angles, movements, lighting types, mood)?',
            'Technical Terminology: Correct use of filmmaking terminology (e.g., close-up, wide shot, three-point lighting, etc.)?',
            'Clarity and Coherence: Is the response well-organized and easy to understand?'
        ]
    
    # 显示配置
    logger.info("=" * 60)
    logger.info(f"Pairwise Model Comparison")
    logger.info("=" * 60)
    logger.info(f"Model A: {args.model_a}")
    logger.info(f"  Predictions: {pred_a_path}")
    logger.info(f"Model B: {args.model_b}")
    logger.info(f"  Predictions: {pred_b_path}")
    logger.info(f"Judge: {args.judge}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Criteria: {len(config['evaluation_criteria'])} criteria")
    logger.info("=" * 60)
    
    # 创建并运行评估器
    evaluator = PairwiseRankingEvaluator(config)
    result = await evaluator.evaluate([], None)
    
    # 保存结果
    results_dir = output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存汇总结果
    summary = {
        'overall_score': result.overall_score,
        'detailed_scores': result.detailed_scores,
        'metadata': result.metadata
    }
    
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # 保存详细判断
    with open(results_dir / 'detailed_judgments.json', 'w') as f:
        json.dump(result.detailed_scores.get('detailed_judgments', []), f, indent=2, default=str)
    
    # 显示结果摘要
    logger.info("\n" + "=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info(f"Total Comparisons: {result.detailed_scores['total_comparisons']}")
    logger.info(f"{args.model_a} Wins: {result.detailed_scores['wins_a']} ({result.detailed_scores['win_rate_a']*100:.1f}%)")
    logger.info(f"{args.model_b} Wins: {result.detailed_scores['wins_b']} ({result.detailed_scores['win_rate_b']*100:.1f}%)")
    logger.info(f"Ties: {result.detailed_scores['ties']} ({result.detailed_scores['tie_rate']*100:.1f}%)")
    logger.info(f"Average Confidence: {result.detailed_scores['avg_confidence']:.3f}")
    logger.info("=" * 60)
    
    # 显示类别分解
    if 'category_breakdown' in result.detailed_scores:
        logger.info("\nCategory Breakdown:")
        for category, stats in sorted(result.detailed_scores['category_breakdown'].items()):
            total = stats['total']
            logger.info(f"\n  {category}:")
            logger.info(f"    {args.model_a}: {stats['wins_a']}/{total} ({stats['wins_a']/total*100:.1f}%)")
            logger.info(f"    {args.model_b}: {stats['wins_b']}/{total} ({stats['wins_b']/total*100:.1f}%)")
            logger.info(f"    Ties: {stats['ties']}/{total} ({stats['ties']/total*100:.1f}%)")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✓ Full results saved to: {results_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())


