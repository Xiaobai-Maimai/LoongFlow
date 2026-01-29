# ML Evolve Agent - 机器学习进化智能体

ML Evolve Agent 是 LoongFlow 框架中专用于机器学习的智能体组件。它采用 PES (Plan-Execute-Summary) 思维范式，通过结构化思考和持续学习来自动构建和优化机器学习解决方案，特别专注于 Kaggle 风格的数据科学竞赛任务。

## 概述

ML Evolve Agent 将进化算法与机器学习专业知识相结合，针对数据科学任务进行了专门优化，具有以下关键特性：

- **全流程自动化**：从数据探索、特征工程到模型训练的完整 ML 工作流
- **竞赛级评估**：支持标准的 Kaggle 风格评估标准和排行榜
- **专业化组件**：内置 evocoder（代码生成器）、evaluator（评估器）等 ML 专用模块
- **多目标优化**：兼顾模型精度、计算效率和泛化能力
- **结构化思考**：基于 PES 范式的规划-执行-总结循环

## 环境准备

确保已安装 Python 3.12+ 并使用 `uv` 进行依赖管理：

```bash
# 在项目根目录执行
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

### MLE-Bench 特定准备

如需运行 MLE-Bench 竞赛，需要额外配置：

```bash
# 初始化 MLE-Bench 环境
./run_mlebench.sh init

# 配置 Kaggle API（用于数据下载）
# 从 https://www.kaggle.com/settings/account 下载 kaggle.json
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

## 任务配置

### 配置文件结构

每个机器学习任务需要创建一个 YAML 配置文件：

```yaml
# 全局目录配置
workspace_path: "./output"

# LLM 配置（支持 OpenAI、Gemini、DeepSeek 等）
llm_config:
  url: "https://your-llm-api/v1"
  api_key: "your-api-key"
  model: "openai/gemini-3-flash-preview"
  temperature: 0.8
  context_length: 128000
  max_tokens: 32768
  top_p: 1.0

# 组件配置（规划器、执行器、总结器）
planners:
  ml_planner:
    react_max_steps: 10
    evo_coder_timeout: 3600

executors:
  ml_executor:
    react_max_steps: 10
    evo_coder_timeout: 86400

summarizers:
  ml_summary:
    react_max_steps: 10

# 进化过程配置
evolve:
  planner_name: "ml_planner"
  executor_name: "ml_executor" 
  summary_name: "ml_summary"
  max_iterations: 100
  target_score: 1.0
  concurrency: 1
  
  evaluator:
    timeout: 1800
    
  database:
    storage_type: "in_memory"
    num_islands: 3
    population_size: 30
    checkpoint_interval: 5
    sampling_weight_power: 1.0
```

### 任务文件结构

机器学习任务需要按照标准结构组织：

```
your_task/
├── task_config.yaml        # 进化和 LLM 配置
├── eval_program.py         # 评分逻辑
├── public/
│   ├── description.md      # 任务描述（智能体可见）
│   ├── train.csv           # 训练数据
│   ├── test.csv            # 测试数据
│   └── sample_submission.csv # 提交格式示例
└── private/
    └── answer.csv          # 真实标签（对智能体隐藏）
```

#### 文件说明

| 文件 | 用途 |
|------|------|
| `description.md` | 任务需求、数据说明和期望输出格式 |
| `train.csv` | 带标签的训练数据 |
| `test.csv` | 无标签的测试数据 |
| `sample_submission.csv` | 预期的提交格式 |
| `answer.csv` | 用于评估的真实标签（智能体不可见） |
| `eval_program.py` | 评分逻辑，返回 0.0-1.0 的分数 |

### 评估代码编写

`eval_program.py` 需要实现标准的评估接口：

```python
def evaluate(task_data_path, best_code_path, artifacts):
    """
    评估函数，返回包含 score 和状态信息的字典
    
    Args:
        task_data_path: 任务数据目录路径
        best_code_path: 最佳代码文件路径
        artifacts: 评估过程参数
        
    Returns:
        dict: 包含状态、分数、指标等信息
    """
    try:
        # 执行解决方案并评估
        result = run_machine_learning_evaluation(best_code_path)
        return {
            "status": "success",
            "score": result["score"],  # 0.0-1.0
            "metrics": {
                "accuracy": result["accuracy"],
                "f1_score": result["f1"]
            },
            "artifacts": {
                "reasoning": "详细评估结果",
                "predictions": result["predictions"]
            }
        }
    except Exception as e:
        return {
            "status": "execution_failed",
            "score": 0.0,
            "summary": f"评估失败: {str(e)}"
        }
```

## 运行流程

### 自定义 ML 任务运行

```bash
# 初始化环境
./run_ml.sh init

# 编辑配置文件（配置 LLM 凭据等）
vim agents/ml_evolve/examples/ml_example/task_config.yaml

# 启动任务（后台运行）
./run_ml.sh run ml_example --background

# 查看实时日志
tail -f ./agents/ml_evolve/examples/ml_example/agent.log

# 停止任务
./run_ml.sh stop ml_example
```

### MLE-Bench 竞赛运行

```bash
# 初始化 MLE-Bench 环境
./run_mlebench.sh init

# 下载竞赛数据
./run_mlebench.sh prepare detecting-insults-in-social-commentary

# 运行进化过程
./run_mlebench.sh run detecting-insults-in-social-commentary --background

# 监控进度
tail -f output/logs/evolux.log
```

### 从检查点恢复

如果任务中断，可以从最近的检查点恢复：

```bash
python agents/ml_evolve/ml_evolve.py \
  --config config.yaml \
  --checkpoint-path ./output/database/checkpoints/checkpoint-checkpoint-iter-50-25
```

## 输出目录结构

执行完成后，`output` 目录将包含以下结构：

```
output/
├── <task-uuid>/                  # 任务唯一标识符
│   └── <iteration-id>/           # 每个进化迭代
│       ├── planner/              # 任务规划阶段
│       │   ├── best_plan.txt     # 最佳规划策略
│       │   └── plan_{编号}.txt   # 详细规划过程
│       ├── executor/             # 执行阶段
│       │   ├── best_solution.py  # 最佳机器学习代码
│       │   └── solution_{编号}.py # 生成的解决方案
│       └── summarizer/           # 总结阶段
│           └── best_summary.txt  # 阶段总结和改进建议
├── database/                     # 进化状态数据库
│   └── checkpoints/
│       └── checkpoint-checkpoint-iter-{迭代数}-{编号}/
│           ├── solutions/        # 所有解决方案的JSON记录
│           ├── best_solution.json # 最佳解决方案信息
│           └── metadata.json     # 元数据（最佳分数、迭代信息等）
├── evaluator/                    # 评估过程记录
│   └── eval_{UUID}/              # 评估过程记录
│       ├── evaluation_result.json # 评估结果
│       └── llm_code_{UUID}.py   # 被评估的机器学习代码
└── logs/                         # 运行时日志
    └── evolux.log
```

### 输出文件说明

- **检查点文件**：保存进化状态，支持断点续跑
- **解决方案文件**：包含生成的机器学习模型代码、超参数、评估分数
- **评估文件**：详细的模型性能评估结果和指标
- **规划文件**：智能体对 ML 任务的思考策略和实验设计
- **总结文件**：阶段性学习总结和后续改进方向

## 可视化监控

ML Evolve Agent 共享通用可视化系统，可监控进化过程：

### 启动可视化服务器

```bash
# 在项目根目录执行
python agents/general_evolve/visualizer/visualizer.py \
  --port 8888 \
  --checkpoint-path output/database/checkpoints
```

### 可视化功能

访问 `http://localhost:8888` 可以看到：

- **🌳 进化树视图**：显示 ML 解决方案的谱系关系
- **📈 分数历史**：展示模型性能随迭代的变化趋势
- **🔍 代码差异**：对比不同版本机器学习代码的修改
- **🗺️ 岛屿地图**：可视化多岛进化保持多样性
- **⚡ 实时更新**：自动刷新显示最新进化状态

## 示例项目

项目提供了丰富的机器学习示例：

### Iris 分类示例 (`ml_example`)

完整的分类任务入门示例：
- Iris 数据集多分类
- 标准化评估流程
- 完整的配置文件模板

### MLE-Bench 竞赛示例

包含多个真实世界的 Kaggle 风格竞赛：

| 类别 | 示例竞赛 | 任务类型 | 难度 |
|------|----------|----------|------|
| 图像分类 | `aerial-cactus-identification` | 仙人掌识别 | 简单 |
| 图像分类 | `dogs-vs-cats-redux-kernels-edition` | 猫狗分类 | 简单 |
| 图像分类 | `histopathologic-cancer-detection` | 癌症检测 | 简单 |
| NLP | `detecting-insults-in-social-commentary` | 侮辱性评论检测 | 简单 |
| 表格数据 | `nomad2018-predict-transparent-conductors` | 材料预测 | 简单 |
| NLP | `google-quest-challenge` | 问答质量评分 | 中等 |
| 表格数据 | `us-patent-phrase-to-phrase-matching` | 专利短语匹配 | 中等 |
| 时间序列 | `predict-volcanic-eruptions-ingv-oe` | 火山爆发预测 | 困难 |
| 生物信息 | `stanford-covid-vaccine` | mRNA 疫苗效力预测 | 困难 |



## 故障排查

### 常见问题

1. **依赖安装问题**
   
    ```bash
    # 确保使用正确的 Python 版本
    python --version  # 应该是 3.12+
    
    # 重新安装依赖
    uv sync
    ```

2. **LLM API 配置错误**
    - 检查 `llm_config` 中的 URL 和 API Key
    - 确认模型名称格式正确（如 `openai/gemini-3-flash-preview`）
    - 验证 API 端点可访问性

3. **数据文件路径错误**
    - 确保数据集文件路径正确
    - 检查 `description.md` 文件是否存在
    - 验证数据文件格式（CSV 格式）

4. **评估超时**
    - 检查 `evaluator.timeout` 设置是否合理
    - 优化评估代码的性能
    - 考虑使用 GPU 加速训练过程

### 调试技巧

- 使用 `--log-level DEBUG` 获取详细日志
- 检查 `output/evaluator/` 目录中的评估记录
- 查看任务目录下的 `agent.log` 文件了解智能体思考过程
- 利用可视化界面分析进化趋势和瓶颈

## 最佳实践

### 任务设计

1. **数据准备**
    - 确保数据格式标准化（CSV 格式，统一编码）
    - 提供清晰的数据字典和特征说明
    - 包含完整的数据预处理指导

2. **评估逻辑**
    - 设计的评估函数要稳定可靠
    - 包含完整的错误处理和边界条件
    - 提供详细的评估指标便于分析

3. **任务描述**
    - 问题定义要清晰明确
    - 期望的输出格式要标准化
    - 包含相关的领域知识和约束条件

### 性能优化

1. **参数调优**
    - 根据任务复杂度合理设置迭代次数
    - 调整并发数以平衡计算资源使用
    - 合理设置超时时间避免资源浪费

2. **资源管理**
    - 监控内存和 GPU 使用情况
    - 对于大规模数据考虑数据分批处理
    - 优化数据加载和处理流水线

### 监控和分析

1. **进度监控**
    - 定期查看日志文件了解智能体思考过程
    - 使用可视化界面跟踪模型性能提升趋势
    - 分析分数变化指导参数调整策略

2. **结果分析**
    - 保存重要检查点便于后续深入分析
    - 比较不同迭代的模型架构差异和性能提升
    - 总结成功策略和失败教训用于后续改进

通过遵循这些指南，你可以充分利用 ML Evolve Agent 的强大能力来解决复杂的机器学习问题，实现从数据探索到模型优化的全流程自动化，在 Kaggle 等机器学习竞赛中取得优异表现。