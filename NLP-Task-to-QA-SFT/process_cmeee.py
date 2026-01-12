"""
CMeEE-V2 Chinese Medical NER Dataset Processing
将 CMeEE-V2 医学命名实体识别数据集转换为 Alpaca 格式的 SFT 训练数据
"""

import json
import os
from collections import Counter
import argparse
from datasets import load_from_disk


# CMeEE 实体类型映射
TYPE_MAPPING = {
    'bod': '身体',           # body
    'dep': '科室',           # department
    'dis': '疾病',           # disease
    'dru': '药物',           # drug
    'equ': '医疗设备',       # equipment
    'ite': '医学检查',       # medical examination item
    'mic': '微生物',         # microorganism
    'pro': '医疗程序',       # medical procedure
    'sym': '临床表现'        # clinical symptom
}


def process_cmeee(
    input_dir: str,
    output_dir: str,
    output_filename: str = "sft_cmeee_train.jsonl"
):
    """
    处理 CMeEE-V2 数据集
    
    Args:
        input_dir: CMeEE 训练集目录（datasets格式）
        output_dir: 输出目录
        output_filename: 输出文件名
    """
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    print("CMeEE-V2 医学NER数据处理")
    
    # 加载数据
    print(f"\n[1/4] 加载数据集: {input_dir}")
    ds = load_from_disk(input_dir)
    print(f"   总数据量: {len(ds)} 条")
    
    # 生成 NER instruction
    all_types = sorted(TYPE_MAPPING.values())
    types_str = "、".join(all_types)
    ner_instruction = f"从以下医学文本中识别所有的医学实体，并按类型分类。实体类型包括：{types_str}"
    
    print(f"\n[2/4] Instruction")
    print(f"   {ner_instruction}")
    
    # 转换数据
    print(f"\n[3/4] 转换为 Alpaca 格式")
    converted = 0
    error_count = 0
    entity_type_counts = Counter()
    
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(ds):
            try:
                text = sample['text']
                entities = sample['entities']
                
                # 按类型组织实体
                entities_by_type = {}
                for entity in entities:
                    entity_type_abbr = entity['type']
                    entity_type_cn = TYPE_MAPPING[entity_type_abbr]
                    entity_text = entity['entity']
                    
                    if entity_type_cn not in entities_by_type:
                        entities_by_type[entity_type_cn] = []
                    entities_by_type[entity_type_cn].append(entity_text)
                    
                    entity_type_counts[entity_type_cn] += 1
                
                # 构建输出
                output_lines = []
                for entity_type in sorted(entities_by_type.keys()):
                    entity_list = "、".join(entities_by_type[entity_type])
                    output_lines.append(f"{entity_type}: {entity_list}")
                
                output = "\n".join(output_lines) if output_lines else "None"
                
                # SFT 格式
                sft_sample = {
                    "instruction": ner_instruction,
                    "input": text,
                    "output": output
                }
                
                f.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")
                converted += 1
                
                if (converted) % 2000 == 0:
                    print(f"   已转换 {converted} 条")
                    
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"   转换失败: {str(e)}")
    
    print(f"\n   转换完成: {converted} 条")
    if error_count > 0:
        print(f"   转换失败: {error_count} 条")
    print(f"   保存路径: {output_path}")
    
    # 实体类型统计
    print(f"\n   实体类型统计:")
    for entity_type, count in sorted(entity_type_counts.items(), 
                                    key=lambda x: x[1], reverse=True):
        print(f"     {entity_type}: {count} 个")
    
    # 更新配置
    print(f"\n[4/4] 更新配置文件")
    dataset_info_path = os.path.join(output_dir, "dataset_info.json")
    
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}
    
    dataset_info["cmeee_sft"] = {
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
            print(f"  输入: {sample['input']}")
            print(f"  输出: {sample['output'][:150]}...")
    
    print("处理完成")
    
    return {
        "total": converted,
        "entity_types": len(entity_type_counts),
        "total_entities": sum(entity_type_counts.values()),
        "output_file": output_path
    }


def main():
    parser = argparse.ArgumentParser(description="处理 CMeEE-V2 数据集")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./cme/train",
        help="输入目录（datasets格式）"
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
        default="sft_cmeee_train.jsonl",
        help="输出文件名"
    )
    
    args = parser.parse_args()
    
    result = process_cmeee(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename
    )
    
    print(f"\n统计信息:")
    print(f"  总样本数: {result['total']}")
    print(f"  实体类型数: {result['entity_types']}")
    print(f"  实体总数: {result['total_entities']}")
    print(f"  输出文件: {result['output_file']}")


if __name__ == "__main__":
    main()