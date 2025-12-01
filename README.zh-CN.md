# MathRAG-Gate

<div align="center">

**资源受限环境下基于置信度门控的鲁棒数学推理 RAG 框架**

[![English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![中文](https://img.shields.io/badge/语言-中文-red)](README.zh-CN.md)

[功能特性](#-功能特性) | [系统架构](#-系统架构) | [实验结果](#-实验结果) | [快速开始](#-快速开始) | [引用](#-引用)

</div>

---

## 📖 项目简介

MathRAG-Gate 是一个专为数学推理任务设计的检索增强生成（RAG）框架，旨在解决传统 RAG 系统在复杂推理任务中的**"相关性-质量鸿沟"**问题。

### 🎯 核心创新

**1. 置信度门控机制（Confidence Gate）**

通过自适应评估规则评分与 LLM 评分的相关性（ρ），动态选择最优的重排序策略，在复杂场景下自动切换至高鲁棒性的 LLM 判官模式。

**2. 分阶段批处理架构（Staged Batching）**

采用时空多路复用设计，将检索/评分与生成阶段解耦，在 12GB 显存环境下实现零溢出（OOM-free）的高效推理，将模型切换开销从 O(N) 降低为 O(1)。

### 🏆 关键成果

- ✅ **准确率提升**：将 Hybrid RRF 的 81.0% 恢复并提升至 **86.8%**
- ✅ **融合噪音过滤**：成功识别并剔除"高语义相关但低逻辑质量"的文档
- ✅ **资源高效**：在消费级硬件（RTX 4070 Ti, 12GB VRAM）上实现 ~4.5s/样本的推理速度
- ✅ **鲁棒性增强**：标准差降低至 σ=0.011，系统稳定性显著提升

---

## 🚨 问题背景

### "相关性-质量鸿沟" (Relevance-Quality Gap)

传统 RAG 系统优化的是**语义相似度**，但在数学推理等对逻辑严谨性要求极高的任务中，这会导致检索到"高语义相关但低逻辑质量"的噪音文档。

**案例**：一篇包含正确答案数字但缺乏推导过程的文档，其向量相似度可能很高，但对 LLM 的思维链（CoT）推理毫无帮助，甚至构成干扰。

### 融合噪音现象 (Fusion Noise)

我们的实验发现，简单地将密集检索与稀疏检索融合（Hybrid RRF），其性能（81.0%）**反而低于单体基线**（86.0%）。这证明了盲目追求高召回率会引入破坏性噪音。

![Performance Comparison](results/exp_20251127_132004/benchmark_results.png)
*图：MathRAG-Gate 有效过滤融合噪音，将准确率从 81% 恢复并提升至 86.8%*

---

## 🏗️ 系统架构

### 整体流程图

```
┌─────────────┐
│  用户查询    │
└──────┬──────┘
       │
       ├─────► 阶段 1: 批量检索与评分 (0.5B 模型)
       │       ├─ 密集检索 (FAISS + BGE)
       │       ├─ 稀疏检索 (BM25)
       │       ├─ 混合融合 (RRF)
       │       └─ 置信度门控 ──┐
       │                       │
       │         ┌─────────────┤
       │         │             │
       │         ▼             ▼
       │    规则-RQP      LLM-RQP (Qwen 0.5B)
       │     (快速)        (鲁棒)
       │         │             │
       │         └───► ρ < 0.45? ──┘
       │                  │
       │         质量感知重排序
       │                  │
       │         ┌────────┴──────────┐
       │         │  Top-K 高质量      │
       │         │  推理模板          │
       │         └───────────────────┘
       │
       ├─────► 阶段 2: 批量生成 (7B 模型)
       │       └─ Qwen-7B 生成器 (思维链推理)
       │
       ▼
┌──────────────┐
│   最终答案    │
└──────────────┘
```

### 关键组件说明

#### 1. 混合检索基座 (Hybrid Retrieval Base)

- **密集检索（Dense Retrieval）**：使用 `BAAI/bge-small-en-v1.5` 模型与 FAISS 向量库
- **稀疏检索（Sparse Retrieval）**：使用 BM25 算法进行关键词匹配
- **混合融合（Hybrid Fusion）**：采用 RRF（倒数排序融合）算法生成 Top-10 候选文档

#### 2. 推理质量评估器 (RQP: Reasoning Quality Predictor)

**Rule-RQP（基线锚点）**

基于结构启发式的快速评分器，评估文档的规范性：

```
Score = w_logic × N_logic + w_struct × N_struct + w_math × N_math + w_box × I_box
```

特征包括：
- 逻辑连接词（"Therefore", "\implies"）
- 结构化标记（"Step 1", "\begin{align}"）
- 数学密度（等式、LaTeX 公式）
- 最终答案标记（`\boxed{}`）

**LLM-RQP（鲁棒判官）**

使用 Qwen2.5-0.5B 模型，通过 Few-Shot 提示进行语义评估：
- 对比锚点（Contrastive Anchors）：低质量示例（Score 1）vs 高质量示例（Score 5）
- 确定性推理控制：Temperature=0.1，Token Limitation=10
- 鲁棒解析：正则表达式提取 `[[score]]` 格式

#### 3. 置信度门控 (Confidence Gate)

**核心机制**：

```
ρ = Pearson(S_rule, S_llm)

Φ = { Φ_Rule(Low Cost),         if ρ > τ
    { Φ_LLM(High Robustness),   if ρ ≤ τ
```

其中 τ = 0.45（阈值）

**科学发现：结构-逻辑正交性**

![Correlation Analysis](results/figure5_correlation_analysis.png)
*图：Rule-RQP 与 LLM-RQP 的相关性分析（ρ=0.139 < 0.45），证明结构规范性与逻辑正确性在 MATH 数据集中呈现显著解耦*

**解释**：
- 在简单任务中，规则评分与逻辑质量高度相关，Gate 选择快速的 Rule 模式
- 在复杂任务（如 MATH 数据集）中，相关性崩溃（ρ < 0.45），Gate 自动触发 LLM Fallback

#### 4. 分阶段批处理 (Staged Batching)

**工程挑战**：12GB VRAM 无法同时承载 7B 生成器和 0.5B 判官

**解决方案**：时空多路复用

- **阶段 1**：仅加载 Embedding + 0.5B 模型，完成所有样本的检索与评分
- **阶段 2**：卸载上述模型，加载 7B Generator，利用缓存的 Top-K 文档批量生成答案

**效果**：
- 模型切换开销从 O(N) 降低为 O(1)
- 平均延迟稳定在 ~4.5s/样本
- 零显存溢出（OOM-free）

---

## 📊 实验结果

### 整体性能对比

| 方法 (Method) | 准确率 (%) | 标准差 (±) | 召回率@5 (%) | 延迟 (秒) |
|--------------|-----------|-----------|--------------|----------|
| 密集检索 | 86.00 | 0.0000 | 100.00 | 4.49 |
| 稀疏检索 (BM25) | 85.00 | 0.0000 | 99.00 | 4.46 |
| **混合检索 (RRF)** | **81.00** ❌ | 0.0000 | 100.00 | 4.59 |
| **MathRAG-Gate (本文)** | **86.80** ✅ | 0.0110 | 100.00 | 4.53 |

*基于 5 轮重复实验的聚合数据（均值 ± 标准差）*

### 关键发现

**1. 融合噪音的证据**

Hybrid RRF 引入的噪音主要表现为**逻辑错误（Logic Errors）**的激增：

- Hybrid RRF：19 个逻辑错误
- MathRAG-Gate：14 个逻辑错误
- **26% 的逻辑错误减少**

**2. 超参数调优的哲学意义**

![Hyperparameter Heatmap](results/optimization_logs/figure6_heatmap.png)
*图：超参数敏感性热力图 - 最优配置为 W_RRF=0.1, W_Quality=0.9*

**分析**：
- 这一极端权重分配表明，系统必须**极度信任内部的质量判断（Quality）**，而几乎完全忽略外部的检索排名（Rank）
- 从数据上验证了本项目 **"质量优于相关性"** 的核心假设

**3. Top-K 质量密度曲线**

MathRAG-Gate 的重排序成功将高质量的"推理模板"推至 Top-1：

- **Hybrid RRF（初始）**：质量分数剧烈震荡（0.75 → 0.45 → 0.65...）
- **MathRAG-Gate（重排序后）**：单调递减的质量密度（0.92 → 0.85 → 0.70...）

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.10+
- **显存**: 最低 12GB（推荐 16GB+）
- **Ollama**: 需预先安装 Qwen2.5 模型

### 1. 安装依赖

```bash
# 克隆项目
git clone <your-repo-url>
cd mathrag_gate_project

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 Ollama 模型

```bash
# 安装生成器模型（7B）
ollama pull qwen2.5:7b

# 安装判官模型（0.5B）
ollama pull qwen2.5:0.5b
```

### 3. 配置环境变量（可选）

创建 `.env` 文件：

```bash
# Hugging Face 镜像（中国用户）
HF_ENDPOINT=https://hf-mirror.com

# API Key（如需使用在线模型）
DASHSCOPE_API_KEY=your_api_key_here
```

### 4. 运行基准测试

```bash
# 单次实验（100 样本）
python run_experiments.py

# 超参数优化（50 样本 × 5 组配置）
python run_optimization.py
```

### 5. 查看结果

```bash
# 实验结果保存在带时间戳的文件夹中
results/
├── exp_20231124_153000/
│   ├── benchmark_results.png       # 可视化图表
│   ├── final_aggregated_report.csv # 聚合统计
│   └── run_*_detailed.csv          # 详细日志
```

---

## ⚙️ 配置说明

### 核心参数 (`src/config.py`)

```python
class Settings(BaseSettings):
    # --- LLM 配置 ---
    LLM_MODEL_NAME: str = "qwen2.5:7b"          # 生成器模型
    QJUDGE_MODEL_NAME: str = "qwen2.5:0.5b"     # 判官模型

    # --- 检索配置 ---
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    DENSE_TOP_K: int = 5
    SPARSE_TOP_K: int = 5
    HYBRID_TOP_K: int = 10

    # --- 重排序权重（经调优）---
    RRF_WEIGHT: float = 0.1      # 检索排名权重
    QUALITY_WEIGHT: float = 0.9  # 质量评分权重

    # --- 置信度门控 ---
    GATE_THRESHOLD: float = 0.45      # 相关性阈值
    GATE_SAMPLE_SIZE: int = 200       # 采样数量
```

### 调优建议

- **资源受限环境**：降低 `GATE_SAMPLE_SIZE` 至 100 以加速初始化
- **更高精度需求**：提升 `QUALITY_WEIGHT` 至 0.95
- **不同领域**：根据 `run_optimization.py` 的结果调整权重

---

## 📁 项目结构

```
mathrag_gate_project/
├── src/
│   ├── config.py                      # 全局配置管理
│   ├── utils/
│   │   └── data_loader.py             # MATH 数据集加载
│   ├── retriever/
│   │   ├── dense_retriever.py         # 密集检索（BGE + FAISS）
│   │   ├── sparse_retriever.py        # 稀疏检索（BM25）
│   │   ├── hybrid_retriever.py        # 混合检索（RRF 融合）
│   │   ├── main_retriever.py          # 主检索器（含 Gate）
│   │   ├── rqar_rule.py               # Rule-RQP 评分器
│   │   ├── rqar_llm.py                # LLM-RQP 评分器
│   │   └── rqar_explainer.py          # 可解释性模块
│   ├── migration/
│   │   └── confidence_gate.py         # 置信度门控核心逻辑
│   ├── eval/
│   │   └── evaluate.py                # 评估框架（分阶段批处理）
│   └── monitoring/
│       └── metrics.py                 # 系统监控与日志
├── run_experiments.py                 # 实验编排脚本（5 轮重复）
├── run_optimization.py                # 超参数网格搜索
├── requirements.txt                   # 依赖列表
├── paper.pdf                          # 项目论文
├── README.md                          # 英文版本
└── README.zh-CN.md                   # 本文档（中文）
```

---

## 🔬 核心技术细节

### 1. Rule-RQP 评分公式

```python
Score = w_logic × N_logic        # 逻辑连接词（30%）
      + w_struct × N_struct      # 结构标记（30%）
      + w_math × N_math          # 数学密度（25%）
      + w_box × I_box            # 最终答案（15%）
```

**特征示例**：
- `N_logic`: "Therefore", "\implies", "\because"
- `N_struct`: "Step 1", "\begin{align}"
- `N_math`: 等式数量、LaTeX 公式（`\frac`, `\sqrt`）
- `I_box`: 检测 `\boxed{}` 标记

### 2. LLM-RQP Few-Shot 提示模板

```
You are a math grader. Rate the reasoning quality from 1 to 5.

### Example 1 (Low Quality - Score 1)
Answer: "The answer is 5."
Reason: No steps, no logic, just a number.
Rating: [[1]]

### Example 2 (High Quality - Score 5)
Answer: "First, let x be the width. Since the area is 20,
         we have x * (x+1) = 20. Solving for x, we get x=4..."
Reason: Clear variables, logical steps, and derivation.
Rating: [[5]]

### Target Answer to Grade
{truncated_text}

Rate from 1 to 5. Output ONLY: [[score]]
```

### 3. RRF 融合算法

```python
def rrf_score(rank_dense, rank_sparse, k=60):
    score = 1.0 / (k + rank_dense + 1) + 1.0 / (k + rank_sparse + 1)
    return score
```

### 4. Gate 决策逻辑

```python
def check_consistency(self, documents):
    # 采样 200 个文档
    sample_docs = random.sample(documents, 200)

    # 计算双评分
    rule_scores = [rule_rqp(doc) for doc in sample_docs]
    llm_scores = [llm_rqp(doc) for doc in sample_docs]

    # Pearson 相关系数
    rho, _ = pearsonr(rule_scores, llm_scores)

    # 决策
    if rho >= 0.45:
        return "USE_RULE"  # 高一致性 → 快速模式
    else:
        return "USE_LLM"   # 低一致性 → 鲁棒模式
```

---

## 📈 实验复现

### 稳定性实验（5 轮重复）

```bash
python run_experiments.py
```

**输出**：
- `results/exp_{timestamp}/final_aggregated_report.csv`
- `results/exp_{timestamp}/benchmark_results.png`

### 超参数网格搜索

```bash
python run_optimization.py
```

**搜索空间**：
```python
search_space = [
    {"rrf": 0.9, "quality": 0.1},  # 主要依赖检索排名
    {"rrf": 0.7, "quality": 0.3},  # 传统设置
    {"rrf": 0.5, "quality": 0.5},  # 平衡模式
    {"rrf": 0.3, "quality": 0.7},  # 偏向质量
    {"rrf": 0.1, "quality": 0.9},  # 极度信任质量 ✅ 最优
]
```

**输出**：
- `results/optimization_logs/optimization_report.csv`
- `results/optimization_logs/tuning_curve.png`

---

## 🎓 引用

如果您在研究中使用了 MathRAG-Gate，请引用：

```bibtex
@article{mathrag_gate_2024,
  title={MathRAG-Gate: A Confidence-Gated RAG Framework for Robust Mathematical Reasoning under Resource Constraints},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

**改进方向**：
1. **神经符号验证器**：引入 Python/SymPy 代码执行模块作为"硬逻辑"验证器
2. **难度感知路由**：基于问题复杂度的动态推理策略
3. **多语言支持**：扩展至中文数学推理任务

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 📮 联系方式

- **作者**: [陈晓腾、叶芮嘉、梁誉瀚]
- **邮箱**: [xchen400@connect.hkust-gz.edu.cn]
- **项目主页**: [GitHub Repository](https://github.com/insensible104/MathRAG-Gate)

---

## 🙏 致谢

- **数据集**: [MATH Dataset](https://github.com/hendrycks/math) by Hendrycks et al.
- **框架**: [LlamaIndex](https://github.com/run-llama/llama_index)
- **模型**: [Qwen2.5](https://github.com/QwenLM/Qwen2.5) by Alibaba Cloud
- **Embedding**: [BGE](https://huggingface.co/BAAI/bge-small-en-v1.5) by BAAI

---

<div align="center">

**🌟 如果这个项目对您有帮助，请给我们一个 Star！🌟**

[![Star History Chart](https://api.star-history.com/svg?repos=your-repo&type=Date)](https://star-history.com/#your-repo&Date)

</div>
