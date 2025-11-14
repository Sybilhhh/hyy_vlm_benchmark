#!/usr/bin/env python3
"""
运行 Test-50 Camera/Lighting Pairwise Ranking 评估

这个脚本用于评估两个模型在摄影机和灯光描述任务上的表现

使用方法:
    python run_camera_ranking.py \\
        --pred-a ./output/qwen3vl_8b/test50_camera/predictions/predictions.json \\
        --pred-b ./output/qwen25vl_7b/test50_camera/predictions/predictions.json \\
        --name-a "Qwen3-VL-8B" \\
        --name-b "Qwen2.5-VL-7B" \\
        --concurrent 5
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluators.test50_ranking import Test50RankingEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Test-50 Camera/Lighting Pairwise Ranking Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_camera_ranking.py \\
      --pred-a ./output/model_a/test50_camera/predictions/predictions.json \\
      --pred-b ./output/model_b/test50_camera/predictions/predictions.json
  
  # With custom names and higher concurrency
  python run_camera_ranking.py \\
      --pred-a ./output/qwen3vl_8b/test50_camera/predictions/predictions.json \\
      --pred-b ./output/qwen25vl_7b/test50_camera/predictions/predictions.json \\
      --name-a "Qwen3-VL-8B" \\
      --name-b "Qwen2.5-VL-7B" \\
      --concurrent 8
        """
    )
    
    parser.add_argument('--pred-a', required=True, 
                       help='Path to model A predictions JSON file')
    parser.add_argument('--pred-b', required=True,
                       help='Path to model B predictions JSON file')
    parser.add_argument('--name-a', default='Model A',
                       help='Display name for model A (default: Model A)')
    parser.add_argument('--name-b', default='Model B',
                       help='Display name for model B (default: Model B)')
    parser.add_argument('--concurrent', type=int, default=5,
                       help='Max concurrent requests (default: 5)')
    parser.add_argument('--frames', type=int, default=8,
                       help='Number of video frames to sample (default: 8)')
    parser.add_argument('--output', '-o', default='./output/camera_ranking',
                       help='Output directory (default: ./output/camera_ranking)')
    
    return parser.parse_args()


async def main():
    args = parse_args()
    
    print("=" * 80)
    print("🎥 Test-50 Camera/Lighting Pairwise Ranking Evaluation")
    print("=" * 80)
    
    # 验证输入文件
    pred_a = Path(args.pred_a)
    pred_b = Path(args.pred_b)
    
    if not pred_a.exists():
        print(f"❌ Error: Model A predictions not found: {pred_a}")
        return 1
    
    if not pred_b.exists():
        print(f"❌ Error: Model B predictions not found: {pred_b}")
        return 1
    
    print(f"\n📁 Input Files:")
    print(f"  Model A: {pred_a}")
    print(f"  Model B: {pred_b}")
    
    # 检查预测文件
    with open(pred_a) as f:
        preds_a_data = json.load(f)
    with open(pred_b) as f:
        preds_b_data = json.load(f)
    
    print(f"\n📊 Data:")
    print(f"  Model A predictions: {len(preds_a_data)}")
    print(f"  Model B predictions: {len(preds_b_data)}")
    
    # 检查是否有视频路径
    has_video = 'video_path' in preds_a_data[0] if preds_a_data else False
    if has_video:
        video_path = preds_a_data[0].get('video_path')
        if video_path and Path(video_path).exists():
            print(f"  ✓ Video-aware evaluation enabled (with actual video content)")
            print(f"    Frames per video: {args.frames}")
            print(f"    GPT-4o will see the video to judge camera/lighting accuracy")
        else:
            print(f"  ⚠️  Video paths in predictions, but files not found")
            print(f"    Falling back to text-only evaluation")
    else:
        print(f"  ℹ️  Text-only evaluation")
    
    print(f"\n⚙️  Configuration:")
    print(f"  Max concurrent requests: {args.concurrent}")
    print(f"  Model A: {args.name_a}")
    print(f"  Model B: {args.name_b}")
    print(f"  Task: Camera and lighting setup description")
    
    # 构建配置
    config = {
        'model_a_predictions': str(pred_a.absolute()),
        'model_b_predictions': str(pred_b.absolute()),
        'model_a_name': args.name_a,
        'model_b_name': args.name_b,
        'judge_model': 'gpt-4o',
        'num_video_frames': args.frames,
        'max_concurrent': args.concurrent,
        'timeout_s': 120,
        'max_retry': 3,
        'retry_backoff_s': 5.0,
        # GPT-4o API 配置
        'model_name': 'gpt-4o',
        'endpoint': 'https://openaieastus2instance.openai.azure.com/openai/deployments/gpt-4o-cv-chx0812/chat/completions?api-version=2025-01-01-preview',
        'api_version': '2024-12-01-preview',
        'api_key': '990a353da7b44bef8466402378c486cd',
        'max_tokens': 4096,
        'temperature': 0,
        'top_p': 1.0,
    }
    
    # 预估时间
    estimated_time_per_sample = 120 / args.concurrent  # 假设每个请求120秒，根据并发调整
    estimated_total = len(preds_a_data) * estimated_time_per_sample
    
    print(f"\n⏱️  Starting evaluation...")
    print(f"  Estimated time: ~{estimated_total/60:.1f} minutes")
    print(f"  (Actual time may vary based on API response time)")
    print()
    
    start_time = time.time()
    
    try:
        evaluator = Test50RankingEvaluator(config)
        result = await evaluator.evaluate([], None)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("✅ Camera/Lighting Evaluation Completed!")
        print("=" * 80)
        
        details = result.details
        
        print(f"\n⏱️  Performance:")
        print(f"  Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
        print(f"  Average per comparison: {elapsed_time / details['total_samples']:.1f}s")
        print(f"  Throughput: {details['total_samples'] / (elapsed_time / 60):.1f} comparisons/min")
        
        print(f"\n📊 Results:")
        print(f"  {args.name_a} wins: {details['wins_a']} ({details['win_rate_a']:.1%})")
        print(f"  {args.name_b} wins: {details['wins_b']} ({details['win_rate_b']:.1%})")
        print(f"  Ties: {details['ties']} ({details['tie_rate']:.1%})")
        print(f"  Errors/Skipped: {details['errors']}")
        print(f"  Total: {details['total_samples']}")
        
        # 按类别统计
        if details.get('category_stats'):
            print(f"\n📁 Results by Video Category:")
            for cat, stats in sorted(details['category_stats'].items()):
                print(f"  {cat:20s}: {args.name_a} {stats['wins_a']:2d}/{stats['total']:2d} " +
                      f"({stats['win_rate_a']:5.1%}) vs {args.name_b} {stats['wins_b']:2d}/{stats['total']:2d} " +
                      f"({stats['win_rate_b']:5.1%})")
        
        # 保存结果
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        with open(output_dir / 'detailed_results.json', 'w') as f:
            json.dump(details, f, indent=2, default=str)
        
        # 保存汇总
        summary = {
            'task': 'test-50-camera-lighting',
            'model_a': args.name_a,
            'model_b': args.name_b,
            'elapsed_time': elapsed_time,
            'avg_per_comparison': elapsed_time / details['total_samples'],
            'throughput': details['total_samples'] / (elapsed_time / 60),
            'wins_a': details['wins_a'],
            'wins_b': details['wins_b'],
            'ties': details['ties'],
            'errors': details['errors'],
            'win_rate_a': details['win_rate_a'],
            'win_rate_b': details['win_rate_b'],
            'tie_rate': details['tie_rate'],
            'concurrent': args.concurrent,
            'video_frames': args.frames,
            'video_aware': has_video,
        }
        
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_dir}/")
        print(f"  - detailed_results.json (full results with individual judgments)")
        print(f"  - summary.json (summary statistics)")
        
        # 速度对比
        sequential_time = details['total_samples'] * 5
        speedup = sequential_time / elapsed_time
        
        print(f"\n🚀 Concurrency Speedup:")
        print(f"  Estimated sequential time: ~{sequential_time/60:.1f} minutes")
        print(f"  Actual concurrent time: {elapsed_time/60:.1f} minutes")
        print(f"  Speedup: {speedup:.1f}x faster!")
        
        # 显示几个示例判断
        print(f"\n📝 Sample Judgments (first 3):")
        for i, judgment in enumerate(details['individual_results'][:3]):
            print(f"\n  {i+1}. {judgment['id']}")
            print(f"     Winner: {judgment['winner']} (confidence: {judgment['confidence']}/5)")
            print(f"     Reason: {judgment['reason'][:80]}...")
        
        return 0
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Error after {elapsed_time:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

