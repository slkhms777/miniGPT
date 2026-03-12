# miniGPT

从零开始手工搭建的自回归大语言模型（Autoregressive LLM）。

> **No Vibe Coding** —— 严格参照 GPT-3、Llama 3 与 Qwen 3 原论文，逐模块实现注意力机制、层归一化、训练循环等核心组件。

## 项目内容

- **纯手搓的实现**：不依赖 Transformers 等高层封装，从最底层矩阵乘法开始构建完整的 GPT 架构
- **现代架构设计**：集成 GQA (Grouped Query Attention) 和 KV Cache 等 engineering 技术  
- **端到端全流程**：涵盖数据构建、预训练、推理部署的完整 pipeline
- **轻量且可验证**：基于 TinyStories 数据集训练，对设备要求较低

## 路线图 (TODO)
- [ ] 增强代码可读性
- [ ] 引入多样性更丰富的训练数据集
- [ ] 加强训练和推理的 infra（如集成 FlashAttention）
- [ ] 尝试引入 SFT（监督微调）和 RL（强化学习）后训练阶段

## 项目结构

```
miniGPT/
├── configs/                   # 配置文件目录
│   └── config.yaml            # 模型、训练以及推理的参数配置
├── utils/                     # 数据处理模块
│   ├── data.py                # Parquet 格式的 Dataset 与 DataLoader 实现
│   └── convert_txt_to_pq.py   # 数据预处理
├── src/                       # 核心模型架构
│   ├── attention.py           # 自注意力机制 (GQA + KV Cache)
│   ├── decoder.py             # Decoder Block
│   ├── gpt.py                 # 主模型 GPT 架构
│   ├── emb.py                 # 词嵌入
│   ├── norm.py                # RMSNorm 
│   ├── mlp.py                 # Llama 3 风格的 SwiGLU MLP 实现
│   ├── tokenizer.py           # 基于 Tiktoken 的 tokenizer
│   └── rope.py                # 旋转位置编码
├── fetch_data.sh              # TinyStories 数据下载与预处理脚本
├── train.py                   # 训练
└── inference.py               # 推理
```


## 快速开始

### 环境配置

本项目推荐使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理（[官方文档](https://docs.astral.sh/uv/)）：

```bash
# 使用 uv 同步环境（推荐）
uv sync
```

**备选方案**：使用 Conda

```bash
conda create -n miniGPT python=3.13
conda activate miniGPT
pip install -r requirements.txt
```

### 数据准备

本项目使用 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 数据集：

```bash
bash scripts/fetch_data.sh
# 数据下载 + 数据清洗
```
如果无法访问 Hugging Face，可以从 [ModelScope](https://www.modelscope.cn/datasets/AI-ModelScope/TinyStories/resolve/master/TinyStories-train.txt)下载：

### 训练

```bash
# 使用 uv
uv run train.py

# 或使用 Conda
python train.py
```

> **提示**：默认配置（batch_size=24, max_seq_len=513）约需 **16GB 显存**，请根据实际硬件调整参数。经测试4090单卡训练一个 epoch 需要约 3h

### 推理生成

```bash
# 使用 uv
uv run inference.py

# 或使用 Conda
python inference.py
```

> **生成建议**：由于数据集多样性有限，建议在 configs/config.yaml 中设置 `temperature ≤ 1`，以牺牲部分多样性换取更合理的输出结果。

### 生成示例

**Prompt:**
```
In a small village
```

**Generated:**
```
In a small village, there was a little girl named Lily. She liked to help her mom in the kitchen. One day, her mom asked her to help her. Lily was excited to help and help her mom.

Together, they cut a big pot of soup on the stove. Lily was so happy and said, "Thank you, mom!" Her mom smiled and said, "You're welcome, Lily. You're so kind."

Lily and her mom sat down to eat the soup. They talked about how happy they were. They were very happy and thanked Lily for helping her. From that day on, Lily always helped her mom with the soup.
```

## 参考

- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) - Brown et al., 2020
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) - Meta AI, 2024
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) - Alibaba Qwen Team, 2025
- [从 MHA 到 MLA 的演进](https://github.com/haukzero/from-mha-to-mla) - 注意力机制优化
