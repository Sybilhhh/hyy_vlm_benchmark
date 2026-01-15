#!/usr/bin/env python3
"""
将 test.json (JSON Lines 格式) 转换为 CSV，排除 video_embedding 列
"""

import json
import csv
import sys
from pathlib import Path

def flatten_value(value):
    """将复杂类型转换为字符串"""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value

def main():
    input_file = Path(__file__).parent / "test.json"
    output_file = Path(__file__).parent / "test.csv"
    
    # 排除的列
    exclude_columns = {"video_embedding"}
    
    print(f"读取文件: {input_file}")
    
    # 首先读取所有数据，收集所有可能的列名
    all_columns = set()
    rows = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 收集所有列名（排除 video_embedding）
                all_columns.update(k for k in data.keys() if k not in exclude_columns)
                # 移除 video_embedding
                data.pop('video_embedding', None)
                rows.append(data)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
            
            if line_num % 1000 == 0:
                print(f"已处理 {line_num} 行...")
    
    print(f"共读取 {len(rows)} 条记录")
    print(f"共 {len(all_columns)} 列")
    
    # 对列名排序，使输出更有规律
    sorted_columns = sorted(all_columns)
    
    # 写入 CSV
    print(f"写入文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted_columns, extrasaction='ignore')
        writer.writeheader()
        
        for row in rows:
            # 将复杂类型转换为 JSON 字符串
            flat_row = {k: flatten_value(v) for k, v in row.items() if k not in exclude_columns}
            writer.writerow(flat_row)
    
    print(f"转换完成! 输出文件: {output_file}")

if __name__ == "__main__":
    main()

