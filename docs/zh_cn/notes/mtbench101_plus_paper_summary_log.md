# MT-Bench-101+ 改造阶段论文素材总结

## 1. 背景与目标
本项目基于 OpenCompass 对 MT-Bench-101 主观评测链路进行扩展，形成了一个可运行的 MT-Bench-101+ 评估框架。改造目标主要包括：

1. 将原有单一评委式主观评测扩展为多 persona judge 协同评测。
2. 引入 meta-judge，对多个用户视角评审结果进行综合。
3. 提升评分输出的结构化程度、提取稳定性与整体可复现性。
4. 支持 4-way 小样本分片运行、补跑与最终汇总。

## 2. 最终评估框架
最终采用的 MT-Bench-101+ 评测框架包含四个核心环节：

1. 被评模型基于多轮对话生成 candidate response。
2. 三个 persona judges 独立评审 candidate response。
3. meta-judge 综合三位 persona judges 的意见，给出最终评分。
4. 后处理模块提取最终评分，并在 task-level 上进行聚合。

三个 persona judges 分别对应以下用户视角：

- Analytical Rigor Judge：强调正确性、推理严谨性、上下文一致性。
- Practical Task Judge：强调任务完成度、可操作性、是否真正帮助用户解决问题。
- Communication Experience Judge：强调表达清晰度、交互体验与用户感受。

meta-judge 的职责不是简单平均分，而是：

- 综合多位评委的证据；
- 识别共识与主要分歧；
- 在任务原始评价标准下输出最终公平评分。

## 3. 关键实现文件
本次改造涉及的主要文件包括：

### 核心实现
- `opencompass/datasets/subjective/mtbench101.py`
- `opencompass/tasks/subjective_eval.py`
- `opencompass/summarizers/subjective/mtbench101.py`
- `opencompass/openicl/icl_evaluator/lm_evaluator.py`
- `opencompass/models/openai_api.py`
- `opencompass/datasets/__init__.py`
- `opencompass/openicl/icl_evaluator/__init__.py`
- `opencompass/openicl/icl_inferencer/__init__.py`
- `opencompass/openicl/icl_retriever/__init__.py`

### 配置文件
- `configs/datasets/subjective/multiround/mtbench101_judge.py`
- `configs/datasets/subjective/multiround/mtbench101_plus_judge.py`
- `configs/eval_subjective_mtbench101.py`
- `configs/eval_subjective_mtbench101_plus.py`
- `configs/eval_subjective_mtbench101_plus_api_small_part0.py`
- `configs/eval_subjective_mtbench101_plus_api_small_part1.py`
- `configs/eval_subjective_mtbench101_plus_api_small_part2.py`
- `configs/eval_subjective_mtbench101_plus_api_small_part3.py`

### 运行脚本
- `tools/run_mtbench101_plus_api_small_4way.py`

### 环境文件
- `environment_opencompass310.yml`

## 4. 解决的关键问题

### 4.1 SubjectiveEval 结果文件定位问题
在分片评测与带下划线数据集缩写场景下，原始 SubjectiveEval 对预测文件与结果文件的定位存在问题。为此修复了：

- 带下划线数据集 `abbr` 的预测/结果文件定位问题；
- 多分片、小样本模式下的结果文件读取逻辑。

### 4.2 已切片数据集重复切片问题
在配置中已经对数据集进行 `test_range` 切片后，eval 阶段仍可能再次切片，导致：

- 空预测列表；
- 结果与分片错位；
- 某些分片无法正确读取预测文件。

为此修复了 SubjectiveEval 对已切片数据集重复切片的问题。

### 4.3 meta judge prompt 冲突问题
这是本次改造中最关键的 prompt 级问题之一。

#### 问题表现
meta-judge 的 system prompt 和 user prompt 对输出格式提出了相互冲突的要求，导致：

- 输出为空；
- 输出格式漂移；
- `Final Rating` 提取失败。

#### 根因
meta-judge 在最初实现中错误复用了普通 judge 的 system prompt 逻辑，导致：

- system 层仍要求普通 judge 风格输出；
- user 层要求 meta-judge 风格的 `Final Rating: [[score]]` 输出。

两层指令冲突使得大量样本无法稳定提取分数。

#### 修复方式
在 `opencompass/datasets/subjective/mtbench101.py` 中重写 `meta_prompt_construct()` 的 system prompt，使 meta-judge 拥有独立且清晰的角色说明，同时保留 task-specific criteria。

#### 修复效果
在局部验证中，meta 输出提取率从：

- `68 / 100`

提升到：

- `92 / 100`

说明 prompt 冲突确实是主要根因。

### 4.4 剩余失败样本分析
在修复 prompt 冲突后，剩余未成功提取的样本主要表现为：

- 输出为空；
- 主要集中在 `GR` 任务；
- prompt 长度较长。

说明问题已从“系统性格式冲突”收敛为“少量长 prompt 边缘样本的空响应问题”。

## 5. 模型尝试与运行策略调整

### 5.1 GLM-4.5-Air
最初使用 `GLM-4.5-Air` 跑通整体链路，但在 meta 阶段存在少量空输出与提取失败。

### 5.2 glm-5.1
曾尝试切换到 `glm-5.1`，但在 infer 阶段多次遇到 API 网络错误（如 `code=1234`），导致推理任务被中断，因此未作为最终稳定方案。

### 5.3 glm-4.6v
最终将 judge / meta-judge 模型调整为 `glm-4.6v`，在复用已有 candidate predictions 的前提下补跑 eval，得到了稳定结果。这一步证明：

- candidate predictions 可以复用；
- judge / meta-judge 模型可以独立升级；
- 只补跑 eval 即可验证评审框架的稳定性。

## 6. 最终实验设置

### 6.1 数据划分
采用 4-way 小样本分片方案，每片 100 条，共 400 条样本：

- part0: `[0:100]`
- part1: `[100:200]`
- part2: `[200:300]`
- part3: `[300:400]`

### 6.2 运行方式
使用并发脚本进行分片调度与汇总：

- `tools/run_mtbench101_plus_api_small_4way.py`

### 6.3 最终稳定评审配置
- 候选答案：复用已生成的 predictions
- persona judges：`glm-4.6v`
- meta-judge：`glm-4.6v`
- 评测方式：3 个 persona judges + 1 个 meta-judge

## 7. 最终实验结果
四个分片最终提取情况如下：

- part0: `100 / 100`
- part1: `98 / 100`
- part2: `99 / 100`
- part3: `100 / 100`

总计：

- 总样本数：`400`
- 成功提取：`397`
- 提取率：`99.25%`

最终 task-level 汇总分数为：

- `GR = 8.9859`
- `IC = 8.2206`

## 8. 本阶段可提炼的论文贡献
本次改造可以提炼出以下几个论文贡献点：

1. 提出并实现了 MT-Bench-101+ 多 persona 主观评测框架。
2. 引入 meta-judge 对多位 persona judges 的意见进行综合，而非简单平均。
3. 修复了原有主观评测链路中的文件定位、重复切片与 prompt 冲突问题。
4. 显著提升了结构化评分的提取成功率与整体稳定性。
5. 构建了可并行运行、可补跑、可汇总的小样本实验流水线。

## 9. 论文中可使用的图表建议
建议优先准备以下图表：

### 图
1. MT-Bench-101+ 评估框架总览图
2. persona-judge 与 meta-judge prompt 结构图
3. 分数提取率提升对比图
4. 多 persona 意见融合示意图

### 表
1. 实验配置表
2. 主结果表（GR / IC / 提取率）
3. 失败案例分析表
4. 单评委与多评委方案对比表

## 10. 阶段性结论
本次 MT-Bench-101+ 改造已从概念验证进入“可运行、可复现、可汇总”的稳定状态。最终系统具备以下特征：

- 主观评测流程可稳定执行；
- 多 persona + meta-judge 设计已经落地；
- 分数提取成功率达到 `99.25%`；
- 已产出可用于论文撰写的最终实验结果。
