# miniLLM Agent — ReAct, Fallback, and GRPO Experiments for Text-to-SQL

## 1. 项目定位

项目已经从“普通 Text-to-SQL SFT 基线”转向 **Agent-first** 路线：

- 模型学会 **多步推理**（Thought → Action → Observation → Answer）
- 通过 **工具调用**（SQL 执行器）获取真实数据库反馈
- 通过 **ReAct SFT** 先学会稳定的 Agent 轨迹格式和纠错模式
- 通过 **GRPO 强化学习** 进一步探索 reward-driven 改进
- 使用 **DeepSpeed ZeRO-2** 进行高效训练

**当前最佳部署方案**不是纯 GRPO，而是：

```text
Direct SQL generation -> execute on real DB -> if failure, fallback to interactive ReAct
```

**一句话总结**：让小模型先学会 Agent 轨迹，再把它部署成“直接生成优先，失败时进入交互纠错”的 Text-to-SQL 系统。

---

## 2. 项目结构

```
miniLLM/
├── agent/                      # ★ Agent 核心模块
│   ├── env.py                  #   SQL 执行环境（沙箱 SQLite）
│   ├── react.py                #   ReAct 格式定义、prompt 模板、轨迹解析
│   ├── reward.py               #   GRPO reward 函数（格式 + 执行 + 正确性）
│   └── data_gen.py             #   用 teacher LLM 生成 ReAct 训练轨迹
│
├── train.py                    # 原始 SQL SFT 训练器
├── train_react_sft.py          # ★ ReAct SFT 训练器 + DeepSpeed
├── train_grpo.py               # ★ GRPO RL 训练器 + DeepSpeed
├── eval.py                     # 原始 SQL 评估
├── eval_agent.py               # ★ Agent 模式评估（单次 / 交互）
│
├── sql_eval.py                 # SQL 评估工具（EM + 执行匹配）
├── prompts.py                  # 原始 prompt 模板
├── service/                    # FastAPI + vLLM 服务
├── quant/                      # 量化工具
└── bench/                      # 吞吐量基准测试

configs/
└── ds_zero2.json               # ★ DeepSpeed ZeRO-2 配置

scripts/
├── gen_react_data.sh           # ★ 生成 ReAct 训练数据
├── train_react_sft.sh          # ★ ReAct SFT 训练
├── train_grpo.sh               # ★ GRPO RL 训练
├── eval_agent.sh               # ★ Agent 评估
├── train.sh                    # 原始 SFT 训练
├── eval.sh                     # 原始评估
└── ...
```

---

## 3. 方法论

### 3.1 三阶段训练流水线

```
Stage 1: ReAct 数据生成
┌─────────────────────────────────────────────────────────┐
│  Spider train 数据集                                     │
│       ↓                                                  │
│  Teacher LLM (默认 gemini-2.5-pro via API) 生成 ReAct 轨迹 │
│       ↓                                                  │
│  用真实 Spider SQLite 替换 Observation（非虚构结果）      │
│       ↓                                                  │
│  30% 的样本含 intentional error → 纠错轨迹               │
│       ↓                                                  │
│  输出: react_trajectories.jsonl                          │
└─────────────────────────────────────────────────────────┘

Stage 2: ReAct SFT
┌─────────────────────────────────────────────────────────┐
│  QLoRA (4-bit NF4) + DeepSpeed ZeRO-2                   │
│  在 ReAct 轨迹上做 SFT                                   │
│  模型学会: Thought → Action → Observation → Answer 格式   │
│  目标: 格式遵从 + 工具使用 + 错误恢复                     │
└─────────────────────────────────────────────────────────┘

Stage 3: GRPO RL
┌─────────────────────────────────────────────────────────┐
│  对每个问题，模型采样 G 条不同的轨迹                      │
│  每条轨迹提取最终 SQL → 在数据库中执行                    │
│  Reward = 格式 + SQL 合法性 + 结构接近度 + 执行部分匹配 + 正确性 │
│  用 Group Relative Policy Optimization 更新策略           │
│  （无需 critic model，单卡可训练）                        │
└─────────────────────────────────────────────────────────┘

**当前结论**：ReAct SFT + fallback 是稳定最优方案；GRPO 目前是实验分支，还没有超过该方案。
```

### 3.2 ReAct 格式

```
Thought: 需要查询 employees 表中薪资最高的人，涉及 ORDER BY + LIMIT。
Action: execute_sql["SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1"]
Observation: name | salary
             -----------
             Alice | 95000
Thought: 查询成功，Alice 薪资最高。
Answer: SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1
```

含错误恢复的轨迹：
```
Thought: 查找薪资最高的员工。
Action: execute_sql["SELECT name FROM employee ORDER BY salary DESC LIMIT 1"]
Observation: Error: no such table: employee
Thought: 表名错误，应该是 employees 而不是 employee。
Action: execute_sql["SELECT name FROM employees ORDER BY salary DESC LIMIT 1"]
Observation: name
             ----
             Alice
Thought: 修正后查询成功。
Answer: SELECT name FROM employees ORDER BY salary DESC LIMIT 1
```

### 3.3 Reward 设计

当前 GRPO reward 不是单一的 0/1 正确性，而是更密的组合信号：

| 分量 | 信号来源 | 意义 |
|------|----------|------|
| format_reward | Thought / Action / Answer 结构 | 维持 ReAct 格式 |
| validity_reward | SQL 是否像合法查询、是否可执行 | 压制格式崩塌和空输出 |
| structure_reward | 与 gold SQL 的 token / clause 重叠 | 提供中间监督 |
| execution_reward | 与 gold 结果的部分匹配 | 对“接近正确”给增量奖励 |
| correctness_reward | 执行结果是否完全一致 | 最终语义正确性 |
| error_penalty | syntax error / no such column 等 | 显式打击 `EX=?` |

**为什么仍然保留 GRPO 实验？**
- 不需要训练额外的 critic/value model → 单卡可行
- 能直接把执行反馈转成 on-policy 更新信号
- 适合研究 reward hacking、格式崩塌和 agent credit assignment 问题

**但当前项目结论也很明确**：
- ReAct SFT 是主干方案
- GRPO 是研究型分支，不是当前最佳交付物

### 3.4 原生 DeepSpeed 训练

**不使用 TRL/HuggingFace Trainer 封装**，直接调用 DeepSpeed 原生 API：

```python
# 核心流程（train_react_sft.py / train_grpo.py）
engine, optimizer, dataloader, _ = deepspeed.initialize(
    model=model,
    model_parameters=[p for p in model.parameters() if p.requires_grad],
    config=ds_config,           # ZeRO stage, optimizer, batch size 等
    training_data=dataset,
)

for batch in dataloader:
    outputs = engine(input_ids=..., attention_mask=..., labels=...)
    loss = outputs.loss
    engine.backward(loss)       # DeepSpeed 管理梯度分片 + 累积
    engine.step()               # 在 gradient_accumulation_boundary 时真正更新

    # 手动 cosine LR scheduling
    if engine.is_gradient_accumulation_boundary():
        new_lr = cosine_lr(step, total_steps, base_lr, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
```

**ZeRO-2 配置**：
- Stage 2 分片 optimizer states + gradients
- 使用 PyTorch 原生 AdamW（`torch_adam: true`），避免 CUDA JIT 依赖
- 手写 cosine annealing with warmup，不依赖 DeepSpeed scheduler

**为什么不用 Stage 3？** 单卡场景无跨卡 all-gather 需求，Stage 3 的 parameter gather 反而增加开销。

**为什么不用 TRL Trainer？** 面试需要讲清 DeepSpeed 每个 API 的作用：`initialize()` 做什么、`backward()` vs `torch.backward()` 的区别、`is_gradient_accumulation_boundary()` 的意义。封装会隐藏这些。

---

## 4. 使用方法

### 4.1 环境准备

```bash
# 安装依赖（含 agent extras）
uv sync
uv pip install deepspeed openai

# 验证
python -c "from miniLLM.agent.env import SQLExecutionEnv; print('OK')"
```

### 4.2 完整训练流水线

```bash
# Step 1: 生成 ReAct 训练数据（需要 teacher API）
API_KEY=sk-xxx NUM_SAMPLES=1000 bash scripts/gen_react_data.sh

# Step 2: ReAct SFT
MODEL=Qwen/Qwen2.5-3B-Instruct MAX_STEPS=500 bash scripts/train_react_sft.sh

# Step 3: GRPO RL（实验性）
MODEL=Qwen/Qwen2.5-3B-Instruct MAX_STEPS=200 bash scripts/train_grpo.sh

# Step 4: 评估 ReAct SFT
ADAPTER=outputs/react-sft bash scripts/eval_agent.sh

# 交互模式（真实 SQL 执行）
ADAPTER=outputs/react-sft INTERACTIVE=1 bash scripts/eval_agent.sh
```

### 4.3 单步评估对比

```bash
# Baseline: 原始模型
python -m miniLLM.eval --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
    --num-samples 50 --with-execution --load-in-4bit

# SQL SFT: 直接 SQL 生成
python -m miniLLM.eval --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
    --adapter-path outputs/sft-qwen2.5-3b-instruct-sql \
    --num-samples 50 --with-execution --load-in-4bit

# Agent: ReAct / fallback / GRPO 实验
python -m miniLLM.eval_agent --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
    --adapter-path outputs/react-sft \
    --num-samples 50 --with-execution --load-in-4bit --interactive
```

---

## 5. 评估指标

| 指标 | 含义 |
|------|------|
| Exact Match (EM) | 标准化后的 SQL 字符串完全匹配 |
| Execution Match (EX) | SQL 执行结果与 gold 一致 |
| Avg Turns | Agent 平均推理轮数 |
| pass@1 | 单次生成正确率 |
| pass@k (纠错后) | 允许 k 轮纠错后的正确率 |

---

## 6. 关键设计决策（面试 FAQ）

**Q: 为什么用 ReAct 而不是 Chain-of-Thought？**
A: ReAct 允许模型与环境交互（执行 SQL 看结果），而 CoT 只是内部推理。对于 Text-to-SQL，能执行验证是核心优势。

**Q: 为什么还保留 GRPO，而不是只做 SFT？**
A: 因为 execution feedback 对 agent 很重要，GRPO 提供了一个把真实执行结果直接转成 on-policy 更新信号的方法。但在当前项目里，它是研究型分支，不是最优交付路线。

**Q: Reward 现在是怎么设计的？**
A: 不是旧版那种 `0.1/0.2/0.7` 三段式了。当前 reward 是 dense 组合：格式、SQL 合法性、结构接近度、执行部分匹配、最终正确性，再叠加对 `syntax error` / `no such column` 的负奖励。这样做是为了减少 reward 稀疏和 `EX=?`。

**Q: 如何避免 Agent 无限循环？**
A: (1) max_turns 硬限制；(2) 已执行的 SQL 记录在 working memory 中避免重复；(3) RL 训练中短轨迹天然获得效率优势（相同 reward 更少 token）。

**Q: DeepSpeed ZeRO Stage 2 vs 3？**
A: Stage 2 分片 optimizer states + gradients，Stage 3 还分片 parameters。单卡场景无跨卡通信需求，Stage 3 的 gather/scatter 开销反而拖慢训练。Stage 2 + CPU offload 是单卡最优选择。

**Q: 为什么 teacher 轨迹要用真实 Observation 替换？**
A: Teacher 模型会"想象"执行结果，但可能与真实 SQLite 行为不一致（如列名、排序、空表）。用真实执行结果替换确保训练数据的 observation 与 RL 阶段一致，避免 distribution shift。

---

## 7. 面试 STAR 叙事框架

### Situation
在 Text-to-SQL 任务上，发现单次 SFT 模型在复杂查询上准确率有限。尤其是涉及多表 JOIN、嵌套子查询时，模型缺乏验证和纠错能力。

### Task
构建一个端到端的 Agent 训练系统：让模型学会调用 SQL 执行器、分析结果、自我修正，并通过 RL 持续优化。

### Action
1. 设计了 ReAct 格式的 Agent 行为范式（Thought→Action→Observation→Answer）
2. 构建了 SQL 执行沙箱环境，作为 tool use 和 reward 信号来源
3. 用 Teacher LLM 生成高质量 ReAct 轨迹（含 30% 错误恢复样本），并用真实执行结果替换
4. 先用 ReAct SFT 教会稳定轨迹和错误恢复，再把 GRPO 作为实验性增量优化
5. DeepSpeed ZeRO-2 + QLoRA，在单张 24GB GPU 上完成全部训练
6. 设计了 dense reward 和 fallback 推理策略，并分析了 GRPO 的失败模式

### Result
- ReAct SFT v2 直接模式达到 `71.7%` EX
- `Direct + Interactive Fallback` 达到 `78.6%` EX，是当前最优部署方案
- GRPO 在当前配置下没有超过 ReAct SFT，但暴露了格式崩塌、reward hacking 和 agent credit assignment 的关键问题
