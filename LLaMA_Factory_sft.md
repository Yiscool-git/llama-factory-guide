# LLaMA-Factory 微调

本文档记录了使用 LLaMA-Factory 在远程服务器上进行 LLM 模型微调的完整流程

## 环境信息（示例）

- 服务器：GPU 服务器
- GPU：NVIDIA GeForce RTX 4090 (24GB 显存)
- 操作系统：Ubuntu 24 (Linux)
- Python：3.12
- 微调框架：LLaMA-Factory

## 一、环境准备与配置

### 1.1 安装并检查当前目录

```bash
# 安装LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# 检查目录
pwd
```

**命令解释：**
- `pwd`：Print Working Directory，显示当前所在的目录路径

**预期输出：**

```
/root/LLaMA-Factory
```

### 1.2 查看 GPU 状态

```bash
nvidia-smi
```

**命令解释：**
- `nvidia-smi`：NVIDIA System Management Interface，显示 GPU 信息和状态
- 用于确认 GPU 是否正常识别，查看显存大小、温度、利用率等

**重要指标说明：**

| 指标 | 说明 | 正常范围 |
|------|------|----------|
| Memory-Usage | 显存使用情况 (已用/总量) | 训练时 < 23000 MiB |
| GPU-Util | GPU 利用率 | 训练时 80-100% |
| Temp | GPU 温度 | 60-85°C |
| Power | 功耗 | 训练时 300-450W |

### 1.3 验证 LLaMA-Factory 安装

```bash
# 查看版本
llamafactory-cli version
```

**命令解释：**

- `llamafactory-cli`：LLaMA-Factory 的命令行工具
- `version`：查看版本信息，验证是否安装成功

**预期输出：**
```
----------------------------------------------------------
| Welcome to LLaMA Factory, version 0.9.4.dev0           |
|                                                        |
| Project page: https://github.com/hiyouga/LLaMA-Factory |
----------------------------------------------------------
```

## 二、理解训练配置文件

### 2.1 查看训练配置模板

```bash
cat examples/train_lora/llama3_lora_sft.yaml
```

**命令解释：**
- `cat`：Concatenate，显示文件内容
- 用于查看 YAML 配置文件的具体内容

### 2.2 配置文件结构说明

训练配置文件分为以下几个部分：

#### Model 部分（模型配置）

```yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| model_name_or_path | 模型路径，可以是 HuggingFace ID 或本地路径 |
| trust_remote_code | 是否信任模型中的自定义代码 |

#### Method 部分（训练方法）

```yaml
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| stage | 训练阶段，sft = Supervised Fine-Tuning (监督微调) |
| do_train | 是否执行训练 |
| finetuning_type | 微调方法，lora = 参数高效微调 |
| lora_rank | LoRA 的秩，越大越精确但显存占用越多 (建议 8-16) |
| lora_target | 应用 LoRA 的目标层，all = 所有可训练层 |

#### Dataset 部分（数据集配置）

```yaml
dataset: identity,alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
```

**参数说明：**

| 参数 | 说明 | 建议值 |
|------|------|--------|
| dataset | 数据集名称，多个用逗号分隔 | 在 data/dataset_info.json 中定义 |
| template | 对话模板，需要与模型匹配 | llama3, qwen, gpt 等 |
| cutoff_len | 最大序列长度 | 1024-2048，越大显存占用越多 |
| max_samples | 使用的最大样本数 | 用于快速实验 |
| overwrite_cache | 是否重新处理数据 | true = 不使用缓存 |
| preprocessing_num_workers | 数据预处理的 CPU 线程数 | 根据 CPU 核心数调整 |
| dataloader_num_workers | 数据加载的 CPU 线程数 | 4-8 |

#### Output 部分（输出配置）

```yaml
output_dir: saves/llama3-8b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| output_dir | 模型保存路径 |
| logging_steps | 每多少步打印一次日志 |
| save_steps | 每多少步保存一次 checkpoint |
| plot_loss | 是否绘制 loss 曲线图 |
| overwrite_output_dir | 是否覆盖已有的输出目录 |
| save_only_model | false = 保存完整 checkpoint (可恢复训练) |
| report_to | 第三方日志工具，none = 不使用 |

#### Train 部分（训练参数）

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
```

**参数说明：**

| 参数 | 说明 | 影响 |
|------|------|------|
| per_device_train_batch_size | 每个设备的 batch size | 显存占用最关键参数 |
| gradient_accumulation_steps | 梯度累积步数 | 实际 batch = batch_size × 累积步数 |
| learning_rate | 学习率 | LoRA 标准值 1e-4 |
| num_train_epochs | 训练轮数 | 完整遍历数据集的次数 |
| lr_scheduler_type | 学习率调度器 | cosine = 余弦衰减 |
| warmup_ratio | 预热比例 | 前 10% 步数学习率从 0 逐渐升高 |
| bf16 | 使用 BF16 混合精度 | 降低显存占用 |
| resume_from_checkpoint | 从 checkpoint 恢复 | null = 从头开始 |

### 2.3 显存占用计算

#### LoRA 微调显存需求

```
显存占用 = 参数量 × (精度÷8) × 1.5

示例：LLaMA 3 8B + BF16 + LoRA
= 8B × (16÷8) × 1.5
= 24GB (理论值)
```

#### 实际显存分配

| 组成部分 | 占用 | 说明 |
|----------|------|------|
| 模型加载 | 固定 ~16GB | 8B 参数 × 2 bytes (BF16) |
| LoRA 参数 | 固定 ~0.5GB | 只训练很小一部分参数 |
| 梯度和优化器 | 固定 ~2GB | 只针对 LoRA 参数 |
| 激活值 | 动态 2-6GB | 取决于 batch_size 和 cutoff_len |
| 框架开销 | 固定 ~1-2GB | PyTorch、CUDA 等 |

#### Batch Size 对显存的影响

```
激活值大小 正比于 batch_size × cutoff_len

示例（LLaMA 3 8B）：
- batch_size=1, cutoff_len=1024: 激活值 ~2GB → 总显存 ~20GB
- batch_size=2, cutoff_len=1024: 激活值 ~4GB → 总显存 ~22GB
- batch_size=4, cutoff_len=1024: 激活值 ~8GB → 总显存 ~26GB (超过 24GB)
```

### 2.4 数据集系统

LLaMA-Factory 使用 `data/dataset_info.json` 管理数据集：

```bash
cat data/dataset_info.json
```

数据集配置格式：

```json
{
  "alpaca_en_demo": {
    "file_name": "alpaca_en_demo.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

**工作流程：**
1. 配置文件中写 `dataset: alpaca_en_demo`
2. LLaMA-Factory 查找 `data/dataset_info.json`
3. 读取 `data/alpaca_en_demo.json` 文件
4. 按照 columns 映射加载数据

## 三、下载模型（以Qwen2.5-3B示例）

在服务器可能无法直接访问 HuggingFace的情况下，我们使用 ModelScope 下载模型。

### 3.1 安装 ModelScope

```bash
pip install modelscope -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### 3.2 下载 Qwen2.5-3B 模型

```bash
python << 'EOF'
from modelscope import snapshot_download

model_dir = snapshot_download(
    'Qwen/Qwen2.5-3B-Instruct',
    cache_dir='./models'
)

print(f"\n Successful！")
print(f"dir_saving: {model_dir}")
EOF
```

**命令解释：**
- `python << 'EOF' ... EOF`：多行 Python 脚本输入
- `snapshot_download`：ModelScope 的模型下载函数
- `cache_dir='./models'`：指定下载到当前目录的 models 文件夹

### 3.3 查看下载的模型

```bash
ls -lh ./models/Qwen/Qwen2___5-3B-Instruct/
```

**命令解释：**
- `ls`：List，列出目录内容
- `-l`：长格式显示（详细信息）
- `-h`：Human-readable，文件大小用易读格式（KB, MB, GB）

**重要文件说明：**

| 文件 | 大小 | 说明 |
|------|------|------|
| config.json | ~1KB | 模型配置 |
| model-00001-of-00002.safetensors | ~3.7GB | 模型权重 1/2 |
| model-00002-of-00002.safetensors | ~2.1GB | 模型权重 2/2 |
| tokenizer.json | ~7MB | 分词器 |
| generation_config.json | <1KB | 生成配置 |

### 3.4 模型绝对路径

```bash
realpath ./models/Qwen/Qwen2___5-3B-Instruct/
```

**命令解释：**

- `realpath`：显示文件或目录的绝对路径
- 将相对路径转换为完整的绝对路径

**输出示例：**

```
/root/LLaMA-Factory/models/Qwen/Qwen2___5-3B-Instruct
```

## 四、准备训练配置

### 4.1 复制配置模板

```bash
cp examples/train_lora/llama3_lora_sft.yaml examples/train_lora/qwen25_3b_lora_sft.yaml
```

**命令解释：**

- `cp`：Copy，复制文件
- 从 LLaMA 3 的配置模板创建 Qwen2.5 的配置

### 4.2 创建训练配置文件

使用以下命令直接创建配置文件：

```bash
cat > examples/train_lora/qwen25_3b_lora_sft.yaml << 'EOF'
### model
model_name_or_path: ./models/Qwen/Qwen2___5-3B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: identity,alpaca_en_demo
template: qwen
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/qwen2.5-3b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
EOF
```

**命令解释：**
- `cat > file << 'EOF' ... EOF`：创建文件并写入多行内容
- `EOF`：End Of File，内容结束标记

**关键修改说明：**

| 参数 | 修改内容 | 原因 |
|------|----------|------|
| model_name_or_path | 改为本地路径 | 如果网络下载失败 |
| template | 改为 qwen | 匹配 Qwen 模型 |
| cutoff_len | 2048 → 1024 | 降低显存压力，更安全 |
| output_dir | 改为 qwen2.5-3b 路径 | 组织输出文件 |

### 4.3 验证配置文件

```bash
cat examples/train_lora/qwen25_3b_lora_sft.yaml
```

检查配置文件内容是否正确，如果 ModelScope 下载时改变了目录名（如用下划线替代点号），需要修改配置路径。

## 五、启动训练

### 5.1 开启 GPU 监控（新终端）

在新的终端窗口中运行：

```bash
watch -n 1 nvidia-smi
```

**命令解释：**
- `watch`：定时重复执行命令
- `-n 1`：每 1 秒刷新一次
- 保持此窗口打开，实时监控 GPU 状态

**监控重点：**
- Memory-Usage：不要超过 23000 MiB
- GPU-Util：应该在 80-100%
- Temp：保持在 85°C 以下

### 5.2 后台启动训练

```bash
nohup llamafactory-cli train examples/train_lora/qwen25_3b_lora_sft.yaml > train.log 2>&1 &
```

**命令解释：**

| 部分 | 说明 |
|------|------|
| nohup | No Hang Up，让程序在后台持续运行，不受终端关闭影响 |
| llamafactory-cli train | LLaMA-Factory 的训练命令 |
| examples/.../qwen25_3b_lora_sft.yaml | 配置文件路径 |
| > train.log | 将标准输出重定向到 train.log 文件 |
| 2>&1 | 将错误输出(2)也重定向到标准输出(1)，即都写入 train.log |
| & | 放到后台运行 |

**输出示例：**
```
[1] 21629
```
数字 21629 是进程 ID (PID)。

### 5.3 查看训练日志

等待 10-20 秒后，查看训练状态：

```bash
tail -n 50 train.log
```

**命令解释：**
- `tail`：显示文件末尾内容
- `-n 50`：显示最后 50 行

**关键信息：**
- 模型加载成功
- 数据集加载成功
- 训练开始，显示 loss 值

### 5.4 实时查看训练进度

```bash
tail -f train.log
```

**命令解释：**
- `tail -f`：Follow 模式，实时追踪文件新增内容
- 按 Ctrl+C 可以退出查看（不影响后台训练）

**训练过程输出示例：**
```
[INFO] ***** Running training *****
[INFO]   Num examples = 1,090
[INFO]   Num Epochs = 3
[INFO]   Total train batch size = 8
[INFO]   Total optimization steps = 411
[INFO]   Number of trainable parameters = 14,966,784

{'loss': 1.2034, 'learning_rate': 2.14e-05, 'epoch': 0.07}
{'loss': 1.1575, 'learning_rate': 4.52e-05, 'epoch': 0.15}
...
```

### 5.5 停止训练（如果需要）

```bash
kill <PID>
```

将 `<PID>` 替换为实际的进程 ID。

查看训练进程：

```bash
ps aux | grep llamafactory
```

**命令解释：**
- `ps aux`：显示所有正在运行的进程
- `|`：管道符，将前一个命令的输出传给下一个命令
- `grep`：搜索包含 llamafactory 的行

## 六、训练完成与结果分析

### 6.1 确认训练完成

训练完成可 Ctrl+C 退出 `tail -f`。

### 6.2 查看训练输出

```bash
ls -lh saves/qwen2.5-3b/lora/sft/
```

**重要文件：**

| 文件 | 大小 | 说明 |
|------|------|------|
| adapter_model.safetensors | ~58MB | LoRA 权重（最重要） |
| adapter_config.json | ~1KB | LoRA 配置 |
| training_loss.png | ~49KB | Loss 曲线图 |
| trainer_state.json | ~8KB | 训练状态 |
| checkpoint-411/ | 目录 | 完整 checkpoint |

### 6.3 查看 Loss 曲线

```bash
ls -lh saves/qwen2.5-3b/lora/sft/training_loss.png
```

将此文件下载到本地查看训练效果。

**良好训练的特征：**
- Loss 持续下降
- 曲线平滑，波动适中
- 最终趋于平稳收敛
- 无明显上升（无过拟合）

### 6.4 查看训练指标

```bash
cat saves/qwen2.5-3b/lora/sft/train_results.json
```

查看详细的训练结果指标。

## 七、LoRA Merge（合并模型）

### 7.1 理解 LoRA Merge

**当前状态：**
- 基础模型：5.8GB（未修改）
- LoRA 适配器：58MB（只记录变化）
- 使用时需要同时加载两者

**Merge 后状态：**
- 合并后的完整模型：5.8GB
- LoRA 权重已整合到基础模型中
- 可以独立使用，不需要分离的文件

### 7.2 创建 Merge 配置文件

```bash
cat > examples/merge_lora/qwen25_3b_merge.yaml << 'EOF'
### Note: DO NOT use quantized model or quantization_bit when merging lora adapters

### model
model_name_or_path: /root/LLaMA-Factory/models/Qwen/Qwen2___5-3B-Instruct
adapter_name_or_path: /root/LLaMA-Factory/saves/qwen2.5-3b/lora/sft
template: qwen
trust_remote_code: true

### export
export_dir: /root/LLaMA-Factory/models/qwen2.5-3b-merged
export_size: 2
export_device: cpu
export_legacy_format: false
EOF
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| model_name_or_path | 基础模型路径（必须是未量化的） |
| adapter_name_or_path | LoRA 权重路径 |
| template | 模型模板 |
| trust_remote_code | 是否信任自定义代码（Qwen 必须为 true） |
| export_dir | 合并后模型保存位置 |
| export_size | 每个文件最大大小（GB），模型会分片保存 |
| export_device | 导出设备，cpu 更稳定 |
| export_legacy_format | 是否使用旧格式，false = 使用新格式 |

### 7.3 执行 Merge

```bash
llamafactory-cli export examples/merge_lora/qwen25_3b_merge.yaml
```

**命令解释：**
- `llamafactory-cli export`：LLaMA-Factory 的导出/合并命令

**Merge 过程：**

1. 加载基础模型
2. 加载 LoRA 适配器
3. 合并权重（W_merged = W + A × B）
4. 保存到 export_dir
5. 按 export_size 分片保存

### 7.4 查看合并后的模型

```bash
ls -lh /root/LLaMA-Factory/models/qwen2.5-3b-merged/
```

**生成的文件：**

| 文件 | 大小 | 说明 |
|------|------|------|
| model-00001-of-00004.safetensors | ~1.9GB | 模型权重分片 1/4 |
| model-00002-of-00004.safetensors | ~1.9GB | 模型权重分片 2/4 |
| model-00003-of-00004.safetensors | ~1.9GB | 模型权重分片 3/4 |
| model-00004-of-00004.safetensors | ~277MB | 模型权重分片 4/4 |
| model.safetensors.index.json | ~35KB | 索引文件（记录参数位置） |
| config.json | ~1.5KB | 模型配置 |
| tokenizer.json | ~11MB | 分词器 |
| Modelfile | ~464B | Ollama 部署配置 |

**总大小：** 5.8GB

## 八、模型推理 (Inference)

训练和合并完成后，需要测试模型的实际表现。LLaMA-Factory 支持多种推理方式。

### 8.1 推理方式概述

```
推理方式                      用途                            适用场景
-----------------------------------------------------------------------------------------
交互式对话 (chat/webchat)    人工测试模型效果                快速验证、演示
批量推理 (batch)             对大数据集生成预测              性能评估、数据标注
API 服务 (api)               提供 API 接口                   生产部署、集成应用
```

### 8.2 推理引擎选择

```yaml
infer_backend: huggingface  # 默认引擎，兼容性好
infer_backend: vllm        # 快速引擎，批量推理速度快 2-5 倍
```

### 8.3 交互式对话测试

#### 方式 1：命令行对话

```bash
llamafactory-cli chat examples/inference/qwen25_3b_lora.yaml
```

#### 方式 2：Web 界面对话（推荐）

```bash
llamafactory-cli webchat examples/inference/qwen25_3b_lora.yaml
```

### 8.4 创建推理配置文件

#### 评估 LoRA 适配器

```bash
cat > examples/inference/qwen25_3b_lora.yaml << 'EOF'
### model
model_name_or_path: /root/LLaMA-Factory/models/Qwen/Qwen2___5-3B-Instruct
adapter_name_or_path: /root/LLaMA-Factory/saves/qwen2.5-3b/lora/sft
template: qwen
finetuning_type: lora
trust_remote_code: true
infer_backend: huggingface
EOF
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| model_name_or_path | 基础模型路径 |
| adapter_name_or_path | LoRA 适配器路径 |
| template | 对话模板，必须与模型匹配 |
| finetuning_type | 微调类型 |
| infer_backend | 推理引擎选择 |

#### 评估合并后的模型

```bash
cat > examples/inference/qwen25_3b_merged.yaml << 'EOF'
### model
model_name_or_path: /root/LLaMA-Factory/models/qwen25-3b-merged
template: qwen
trust_remote_code: true
infer_backend: huggingface
EOF
```

**注意：** 合并后的模型不需要指定 adapter_name_or_path 和 finetuning_type。

### 8.5 批量推理（使用 vLLM）

当需要在大量数据上生成预测时，使用 vLLM 引擎可以大幅提升速度。

```bash
python scripts/vllm_infer.py \
    --model_name_or_path /root/LLaMA-Factory/models/qwen25-3b-merged \
    --dataset alpaca_en_demo
```

**命令解释：**
- `scripts/vllm_infer.py`：vLLM 批量推理脚本
- `--model_name_or_path`：模型路径（必须是合并后的完整模型）
- `--dataset`：要推理的数据集名称

**限制：**
- vLLM 只支持完整模型，不支持 LoRA 适配器
- 需要使用合并后的模型

### 8.6 API 服务部署

启动 API 服务，提供 OpenAI 兼容的接口。

## 九、模型评估 (Evaluation)

评估模型性能是微调流程的重要环节，可以量化衡量模型的改进效果。

### 9.1 评估方式概述

```
评估类型                      评估内容                        输出指标
-----------------------------------------------------------------------------------------
NLG 评估                     生成质量                        BLEU, ROUGE
通用能力评估                  标准任务准确率                  准确率 (Accuracy)
```

### 9.2 NLG 评估（生成质量评估）

评估模型生成文本的质量，计算 BLEU 和 ROUGE 分数。

#### 创建评估配置文件

```bash
cat > examples/extras/nlg_eval/qwen25_3b_lora_predict.yaml << 'EOF'
### model
model_name_or_path: /root/LLaMA-Factory/models/Qwen/Qwen2___5-3B-Instruct
adapter_name_or_path: /root/LLaMA-Factory/saves/qwen2.5-3b/lora/sft
trust_remote_code: true

### method
stage: sft
do_predict: true
finetuning_type: lora

### dataset
eval_dataset: identity,alpaca_en_demo
template: qwen
cutoff_len: 1024
max_samples: 50
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/qwen2.5-3b/lora/predict
overwrite_output_dir: true

### eval
per_device_eval_batch_size: 1
predict_with_generate: true
ddp_timeout: 180000000
EOF
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| do_predict | 设为 true 执行预测 |
| eval_dataset | 评估数据集 |
| max_samples | 评估样本数量（可调整） |
| output_dir | 预测结果保存路径 |
| predict_with_generate | 使用生成模式 |

#### 执行 NLG 评估

```bash
llamafactory-cli train examples/extras/nlg_eval/qwen25_3b_lora_predict.yaml
```

**命令解释：**
- 虽然命令是 train，但 `do_predict: true` 会执行预测而非训练

#### 查看评估结果

```bash
cat saves/qwen2.5-3b/lora/predict/all_results.json
```

**典型输出示例：**

```json
{
    "predict_bleu-4": 51.66,
    "predict_rouge-1": 62.09,
    "predict_rouge-2": 42.22,
    "predict_rouge-l": 55.09,
    "predict_runtime": 200.94,
    "predict_samples_per_second": 0.498
}
```

### 9.3 通用能力评估

在标准测试集（MMLU、CEVAL、CMMLU）上评估模型的通用能力。

#### 创建通用能力评估配置

```bash
cat > examples/train_lora/qwen25_3b_lora_eval.yaml << 'EOF'
### model
model_name_or_path: /root/LLaMA-Factory/models/Qwen/Qwen2___5-3B-Instruct
adapter_name_or_path: /root/LLaMA-Factory/saves/qwen2.5-3b/lora/sft
trust_remote_code: true

### method
finetuning_type: lora

### dataset
task: ceval_validation
template: fewshot
lang: zh
n_shot: 5

### output
save_dir: saves/qwen2.5-3b/lora/eval

### eval
batch_size: 4
EOF
```

**参数说明：**

| 参数 | 说明 | 选项 |
|------|------|------|
| task | 评估任务 | mmlu_test, ceval_validation, cmmlu_test |
| template | 提示模板 | fewshot（少样本学习） |
| lang | 语言 | en（英文），zh（中文） |
| n_shot | 少样本数量 | 通常为 5 |
| save_dir | 结果保存路径 | 必须指定 |

#### 执行通用能力评估

```bash
llamafactory-cli eval examples/train_lora/qwen25_3b_lora_eval.yaml
```

#### 查看评估结果

```bash
cat saves/qwen2.5-3b/lora/eval/results.json
```

## 十、完整工作流程总结

### 10.1 标准 LLM 微调流程

```
阶段 1: 准备
├─ 环境配置
├─ 模型下载
└─ 理解配置文件

阶段 2: 训练
├─ 创建训练配置
├─ 启动训练
└─ 监控训练过程

阶段 3: 合并
├─ 创建合并配置
├─ 执行 LoRA Merge
└─ 验证合并结果

阶段 4: 评估
├─ 交互式测试（可选）
├─ NLG 评估
└─ 通用能力评估（可选）

阶段 5: 部署
├─ API 服务
├─ 批量推理
└─ 模型分享
```

### 10.2 关键文件清单

#### 训练相关

```
配置文件:
- examples/train_lora/qwen25_3b_lora_sft.yaml

输出文件:
- saves/qwen2.5-3b/lora/sft/adapter_model.safetensors
- saves/qwen2.5-3b/lora/sft/training_loss.png
- saves/qwen2.5-3b/lora/sft/trainer_state.json
```

#### 合并相关

```
配置文件:
- examples/merge_lora/qwen25_3b_merge.yaml

输出文件:
- models/qwen25-3b-merged/model-00001-of-00004.safetensors
- models/qwen25-3b-merged/model-00002-of-00004.safetensors
- models/qwen25-3b-merged/model-00003-of-00004.safetensors
- models/qwen25-3b-merged/model-00004-of-00004.safetensors
```

#### 评估相关

```
配置文件:
- examples/extras/nlg_eval/qwen25_3b_lora_predict.yaml
- examples/train_lora/qwen25_3b_lora_eval.yaml

输出文件:
- saves/qwen2.5-3b/lora/predict/all_results.json
- saves/qwen2.5-3b/lora/eval/results.json
```

## 附录：常用命令参考

### 文件和目录操作

```bash
# 查看当前目录
pwd

# 列出文件
ls -lh <目录或文件>

# 查看文件内容
cat <文件>

# 显示文件开头
head -n 10 <文件>

# 显示文件末尾
tail -n 10 <文件>

# 实时查看文件更新
tail -f <文件>

# 复制文件
cp <源文件> <目标文件>

# 移动/重命名文件
mv <源> <目标>

# 删除文件或目录
rm -rf <文件或目录>

# 创建目录
mkdir -p <目录>

# 查看磁盘使用情况
df -h

# 查看目录大小
du -sh <目录>

# 获取绝对路径
realpath <相对路径>
```

### 进程管理

```bash
# 查看所有进程
ps aux

# 查看特定进程
ps aux | grep <关键词>

# 终止进程
kill <PID>

# 强制终止进程
kill -9 <PID>

# 后台运行命令
nohup <命令> > log.txt 2>&1 &
```

### GPU 监控

```bash
# 查看 GPU 状态
nvidia-smi

# 实时监控 GPU（每秒刷新）
watch -n 1 nvidia-smi

# 持续监控 GPU（每秒刷新）
nvidia-smi -l 1

# 查看特定 GPU 信息
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
```

### LLaMA-Factory 命令

```bash
# 查看版本
llamafactory-cli version

# 查看帮助
llamafactory-cli --help

# 训练模型
llamafactory-cli train <配置文件.yaml>

# 导出/合并模型
llamafactory-cli export <配置文件.yaml>

# 启动 Web UI
llamafactory-cli webui
```
