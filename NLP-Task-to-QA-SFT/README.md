# 中文NLP任务转问答格式的SFT实验

## 项目简介

基于 LLaMa-Factory 将传统NLP任务（文本分类和NER）转换为问答格式，使用Qwen2.5-3B进行监督微调(SFT)

## 实验任务

- **文本分类**：THUNews 中文新闻分类
- **命名实体识别**：CMeEE - v2 医学实体识别

## 数据集

### THUNews

- 来源：cnews数据集
- 规模：50,000条训练数据
- 类别：体育、财经、房产等

### CMeEE - v2

- 来源：CBLUE医学NLP基准
- 规模：15,000条训练数据
- 类别：疾病、药物、症状等

## 数据转换

### 分类任务

原始格式 → 问答格式 

```json
{  
  "instruction": "请对下面的一段新闻文本进行分类...",  
 "input": "新闻内容",  
 "output": "体育" 
}
```

### NER任务

原始格式 → 问答格式

```json
{
  "instruction": "从以下医学文本中识别所有的医学实体...",
  "input": "患者诉头痛、恶心",
  "output": "临床表现: 头痛、恶心"
}
```

## 使用方法

### 数据处理

```bash
python process_thunews.py
python process_cmeee.py
```

### SFT训练

```bash
llamafactory-cli train configs/qwen25_3b_lora_sft.yaml
```

### 合并LoRA

```bash
llamafactory-cli export configs/qwen25_3b_merge.yaml
```







