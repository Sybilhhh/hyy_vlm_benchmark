#!/usr/bin/env python3
"""
检查哪些视频触发了内容过滤器
"""
import json
from pathlib import Path

def check_filtered_videos(predictions_file: str):
    """检查预测结果中哪些视频被内容过滤器拦截"""
    
    if not Path(predictions_file).exists():
        print(f"预测文件不存在: {predictions_file}")
        return
    
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    filtered_videos = []
    successful_videos = []
    
    for pred in predictions:
        video_id = pred.get('id')
        prediction = pred.get('prediction', '')
        video_path = pred.get('video_path', '')
        
        if 'Content filtered' in prediction or 'content_filter' in prediction.lower():
            filtered_videos.append({
                'id': video_id,
                'path': video_path,
                'prediction': prediction
            })
        else:
            successful_videos.append(video_id)
    
    print("=" * 80)
    print("内容过滤器检查结果")
    print("=" * 80)
    
    if filtered_videos:
        print(f"\n⚠️  被过滤的视频 ({len(filtered_videos)} 个):")
        print("-" * 80)
        for vid in filtered_videos:
            print(f"  ID: {vid['id']}")
            print(f"  路径: {vid['path']}")
            print(f"  响应: {vid['prediction'][:100]}")
            print()
    else:
        print("\n✓ 没有视频被内容过滤器拦截")
    
    print(f"\n✓ 成功处理的视频: {len(successful_videos)} 个")
    print("=" * 80)

if __name__ == "__main__":
    predictions_file = "/home/dyvm6xra/dyvm6xrauser04/yuyang/vlm_benchmark/results/predictions/predictions.json"
    check_filtered_videos(predictions_file)


