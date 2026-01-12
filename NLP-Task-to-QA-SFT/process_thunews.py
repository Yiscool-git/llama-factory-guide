"""
THUNews Chinese News Classification Dataset Processing
将 cnews 数据集转换为 Alpaca 格式的 SFT 训练数据
"""

import json
import os
from collections import Counter
import argparse


def process_thunews(
    input_file: str,
    output_dir: str,
    output_filename: str = "sft_thunews_train.jsonl",
    max_length: int = 512
):
    """
    处理 THUNews 数据集
    
    Args:
        input_file: cnews 训练集文件路径
        output_dir: 输出目录
        output_filename: 输出文件名
        max_length: 文本最大长度
    """
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    print("THUNews 数据处理")
    
    # 读取训练集
    print(f"\n[1/4] 读取训练集: {input_file}")
    data_list = []
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split('\t')
                
                if len(parts) != 2:
                    error_count += 1
                    continue
                
                label = parts[0]
                text = parts[1]
                
                if not label or not text:
                    error_count += 1
                    continue
                
                data_list.append({
                    'text': text,
                    'label': label
                })
                
                if len(data_list) % 10000 == 0:
                    print(f"   已读取 {len(data_list)} 条")
                    
            except Exception as e:
                error_count += 1
                if error_count <= 10:
                    print(f"   跳过第 {idx} 行: {str(e)[:50]}")
    
    print(f"\n   成功读取: {len(data_list)} 条")
    if error_count > 0:
        print(f"   跳过异常: {error_count} 条")
    
    # 统计类别
    print(f"\n[2/4] 类别统计")
    label_counts = Counter([item['label'] for item in data_list])
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count} 条")
    
    # 生成 instruction
    all_labels = sorted(set([item['label'] for item in data_list]))
    labels_str = "、".join(all_labels)
    cls_instruction = f"请对下面的一段新闻文本进行分类。可选的类别包括：{labels_str}。请直接输出类别名称。"
    
    # 转换为 SFT 格式
    print(f"\n[3/4] 转换为 Alpaca 格式")
    converted = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data_list:
            sft_sample = {
                "instruction": cls_instruction,
                "input": item['text'][:max_length],
                "output": item['label']
            }
            f.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")
            converted += 1
    
    print(f"   转换完成: {converted} 条")
    print(f"   保存路径: {output_path}")
    
    # 更新 dataset_info.json
    print(f"\n[4/4] 更新配置文件")
    dataset_info_path = os.path.join(output_dir, "dataset_info.json")
    
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}
    
    dataset_info["thunews_sft"] = {
        "file_name": output_filename,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }
    
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"   配置已更新: {dataset_info_path}")
    
    # 查看样例
    print("转换结果预览 (前3条)")
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            sample = json.loads(line)
            print(f"\n第 {i+1} 条:")
            print(f"  类别: {sample['output']}")
            print(f"  内容: {sample['input'][:100]}...")
    
    print("处理完成")
    
    return {
        "total": len(data_list),
        "categories": len(label_counts),
        "output_file": output_path
    }


def main():
    parser = argparse.ArgumentParser(description="处理 THUNews 数据集")
    parser.add_argument(
        "--input",
        type=str,
        default="./cnews/cnews.train.txt",
        help="输入文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="输出目录"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="sft_thunews_train.jsonl",
        help="输出文件名"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="文本最大长度"
    )
    
    args = parser.parse_args()
    
    result = process_thunews(
        input_file=args.input,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        max_length=args.max_length
    )
    
    print(f"\n统计信息:")
    print(f"  总样本数: {result['total']}")
    print(f"  类别数: {result['categories']}")
    print(f"  输出文件: {result['output_file']}")


if __name__ == "__main__":
    main()