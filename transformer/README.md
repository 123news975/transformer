# 英汉翻译器——从零开始实现 Transformer

## 目录
- [1. 项目简介](#1-项目简介)
- [2. 代码库结构](#2-代码库结构)
- [3. 复现指南](#3-复现指南)
  - [3.1. 环境设置](#31-环境设置)
  - [3.2. 数据准备](#32-数据准备)
  - [3.3. 模型训练](#33-模型训练)
  - [3.4. 推理](#34-推理)
- [4. 实验环境与预期结果](#4-实验环境与预期结果)
- [5. 核心发现与总结](#5-核心发现与总结)

## 1. 项目简介
本项目旨在使用 PyTorch 从零开始、不依赖任何高级封装库来完整实现经典的 Transformer 模型，实现**英汉机器翻译**任务。通过该项目，您可以深入理解 Transformer 的每一个核心组件，包括其内部的数据流、掩码机制以及完整的训练与推理流程。

具体结构如下：


- 标准的 Encoder-Decoder 架构，完全基于自注意力和前馈网络。
- **核心组件清晰实现**:
  - `MultiHeadAttention` (多头注意力机制)
  - `ScaledDotProductAttention` (缩放点积注意力)
  - `PositionalEncoding` (位置编码)
  - `PositionwiseFeedForward` (逐位置前馈网络)
- **详细的掩码机制**: 包含了对源语言 Padding 和目标语言 Padding + Subsequent (未来词) 的精确掩码处理。
- **完整的训练与评估流程**: 提供了 `train.py` 脚本，包括数据加载、模型训练、验证循环和损失可视化。
- **交互式推理**: 提供了 `inference.py` 脚本，可以加载训练好的模型，并从命令行进行交互式英汉翻译。
- **配置化管理**: 通过 `config/base.yaml` 文件管理所有超参数，便于调试和复现。

## 2. 代码库结构

```
transformer/
├── .idea/                  # PyCharm IDE 配置文件夹
├── config/
│   └── base.yaml           # 超参数配置文件
├── data/
│   └── cmn.txt             # 英汉平行语料数据文件
├── data.py                 # 数据集类、词典构建和数据处理函数
├── inference.py            # 交互式翻译脚本
├── model.py                # Transformer 模型所有核心组件的定义
├── README.md               # 本文档
├── requirements.txt        # 项目依赖
└── train.py                # 模型训练主脚本
```

### 文件说明

- **`model.py`**: 定义了构成 Transformer 的所有 `nn.Module` 类，包括 `Transformer`, `Encoder`, `EncoderLayer`, `Decoder`, `DecoderLayer`, `MultiHeadAttention` 等。这是项目的核心。
- **`data.py`**: 定义了 `TranslationDataset` 数据集类、用于构建词典的 `build_vocab` 函数以及用于动态填充的 `collate_fn` 函数。
- **`train.py`**: 负责执行模型的整个训练过程。它会加载数据、构建词典、初始化模型、定义优化器和损失函数，并运行训练和验证循环。
- **`inference.py`**: 一个独立的脚本，用于加载训练好的模型权重和词典文件，并启动一个命令行界面，让用户可以输入英文句子并获得中文翻译结果。
- **`requirements.txt`**: 列出了运行本项目所需的所有 Python 库。
- **`config/base.yaml`**: 存储所有超参数，如学习率、批次大小、模型维度等，使训练配置与代码分离。
- **`data/cmn.txt`**: 存放英汉平行语料的文本文件。

## 3. 复现指南

### 3.1. 环境设置


建议使用 PyCharm 打开项目，它会自动识别并建议设置 Python 解释器。

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/123news975/transformer.git
    cd transformer
    ```

2.  **创建并激活虚拟环境**:
    PyCharm 通常会自动处理此步骤。如果需要手动操作：
    ```bash
    # 在项目根目录下
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    
    # macOS / Linux
    source venv/bin/activate
    ```

3.  **安装依赖项**:
    打开 PyCharm 的终端（Terminal）面板，运行以下命令：
    ```bash
    pip install -r requirements.txt
    ```
    > **注意**: `requirements.txt` 中可能指定了特定 CUDA 版本的 PyTorch。请根据您的硬件（是否有 NVIDIA GPU）调整 `torch` 的安装命令。

### 3.2. 数据准备

本项目默认使用 `data/cmn.txt` 作为数据源。请确保该文件存在且格式正确。

**文件格式**应为每行一个翻译对，英文和中文之间用**制表符 `\t`** 分隔。

**`data/cmn.txt` 示例:**
```
Hi.	嗨。
Run!	快跑！
I'm home.	我回来了。
He is ill.	他生病了。
```

### 3.3. 模型训练

1.  **添加模型保存代码**
    为了让 `inference.py` 能够使用训练好的模型，您需要在 `train.py` 脚本的**末尾**（`plt.show()` 之后）添加以下代码：

    ```python
    # train.py (在文件末尾添加)
    
    import pickle
    
    # 定义保存路径
    MODEL_SAVE_PATH = "transformer_scratch.pth"
    VOCAB_SAVE_PATH = "vocab.pkl"
    
    # 1. 保存模型权重
    print("\nTraining complete. Saving model and vocab...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # 2. 保存词典和分词器信息
    vocab_info = {
        "en_vocab": en_vocab,
        "zh_vocab": zh_vocab,
        "tokenizer_en": "basic_english",
        # 记录中文分词方式，便于推理时复现
        "tokenizer_zh_info": "lambda x: list(x)" 
    }
    with open(VOCAB_SAVE_PATH, "wb") as f:
        pickle.dump(vocab_info, f)
    
    print(f"✅ 模型已保存到: {MODEL_SAVE_PATH}")
    print(f"✅ 词典已保存到: {VOCAB_SAVE_PATH}")
    ```

2.  **运行训练**
    在 PyCharm 中，右键点击 `train.py` 文件，然后选择 `Run 'train'`。或者在终端中运行：
    ```bash
    python train.py
    ```
    训练结束后，您会在项目根目录下看到 `transformer_scratch.pth` 和 `vocab.pkl` 两个文件。

### 3.4. 推理

模型训练完成后，即可进行实时翻译。在 PyCharm 中右键点击 `inference.py` 文件，选择 `Run 'inference'`，或者在终端中运行：

```bash
python inference.py
```

脚本会加载 `transformer_scratch.pth` 和 `vocab.pkl`，然后进入交互模式。

**交互示例:**
```
✅ 模型加载成功，进入交互模式
🌍 进入交互式翻译模式 (输入 'quit' 退出)

请输入英文句子: I love machine learning.
🗣️ 翻译结果: 我爱机器学习。

请输入英文句子: How are you?
🗣️ 翻译结果: 你好吗?

请输入英文句子: quit
👋 已退出翻译模式。```
```
## 4. 实验环境与预期结果
本项目在 Google Colab 云端环境中进行开发与训练，具体的软硬件配置如下：

硬件环境:\
平台: Google Colab\
GPU: NVIDIA Tesla T4 (显存 16GB)\
RAM: 12GB\
软件环境:\
语言: Python 3.x\
核心框架: PyTorch (1.11.0+cu113 或更高版本)\
主要库:\
torchtext: 用于英文分词和词典构建。\
pandas: 用于数据加载和初步处理。\
scikit-learn: 用于数据集的划分。\
matplotlib: 用于绘制训练过程中的损失曲线。\
tqdm: 用于显示训练进度的进度条。

预期运行时间:
在完整的数据集上进行训练，每个轮次（epoch）的平均耗时约为 11s 。


预期最终性能:
成功运行 train.py 和 inference.py 脚本后，您应观察到以下结果：\
模型训练:在训练过程中，终端会按轮次（Epoch）打印出训练集损失 (Train Loss) 和验证集损失 (Val Loss)。
随着训练的进行，这两个损失值应呈现稳步下降的趋势，表明模型正在有效地学习。训练全部结束后，会自动弹出一个由 matplotlib 生成的损失曲线图，直观地展示训练过程中的收敛情况。\
产出文件:
训练完成后，项目根目录下会生成两个关键文件：\
transformer_scratch.pth: 保存了训练好的模型权重。\
vocab.pkl: 保存了英文和中文字典，用于后续的推理。

交互式翻译:\
运行 inference.py 脚本后，程序会成功加载模型和词典。\
在提示符后输入一句英文，模型能够生成基本合理、符合语法的中文翻译。由于模型规模和数据集较小，翻译结果可能在流畅度和准确性上无法与商业级翻译引擎媲美，但应能体现出模型已掌握了基本的词汇对齐和句子结构转换能力。


## 5. 核心发现与总结
本项目成功地从零开始实现了一个用于英汉翻译的 Transformer 模型。未来的改进方向可以包括：
扩展数据集: 使用更大规模的平行语料（如 WMT 数据集）进行训练，以提升模型的翻译质量和泛化能力。
优化分词: 采用更先进的分词方法，如 BPE (Byte Pair Encoding)，来更好地处理未登录词和复杂词汇。
引入评估指标: 实现 BLEU 分数等机器翻译领域的标准化评估指标，对模型性能进行量化分析。
超参数调优: 系统地进行超参数搜索，找到更优的模型配置和训练策略（如使用学习率调度器）。

