# MT-Bench-101+ 改造阶段开发时间线日志

## 1. 项目目标
围绕 OpenCompass 中的 MT-Bench-101 主观评测链路，构建一个扩展版的 MT-Bench-101+ 评估框架，核心目标包括：

- 引入多 persona judges；
- 引入 meta-judge 综合多评委意见；
- 修复主观评测链路中实际运行暴露出的工程问题；
- 支持多分片实验与最终结果汇总。

---

## 2. 早期阶段：理解数据流与提示词结构
### 2.1 梳理评测流程
首先对 MT-Bench-101 的现有实现进行梳理，确认主观评测链路涉及：

- 数据集读取；
- prompt 构造；
- 被评模型推理；
- judge 模型评审；
- 分数提取与汇总。

同时梳理了以下核心文件：

- `opencompass/datasets/subjective/mtbench101.py`
- `configs/datasets/subjective/multiround/mtbench101_plus_judge.py`
- `configs/eval_subjective_mtbench101_plus.py`
- 各分片配置文件

### 2.2 梳理输入输出字段
在对话中进一步分析了每个环节中输入、输出字段的组成，以及各字段来源文件，为后续精准修改链路奠定基础。

---

## 3. 中期阶段：扩展多 persona judge 与 meta-judge
### 3.1 引入 persona panel
在原始 MT-Bench-101 的基础上，引入三个 persona judges：

- Analytical Rigor Judge
- Practical Task Judge
- Communication Experience Judge

三者共享原始任务评分标准，但在关注重点上进行差异化设置。

### 3.2 引入 meta-judge
设计 meta-judge，用于读取：

- 候选答案；
- 三位 persona judges 的意见；
- 任务评分标准；

并输出：

- 共识；
- 分歧；
- 最终 justification；
- 最终评分。

### 3.3 构建新配置
新增 MT-Bench-101+ 的主配置与多分片配置，支持小样本分片实验：

- `configs/eval_subjective_mtbench101_plus.py`
- `configs/eval_subjective_mtbench101_plus_api_small_part0.py`
- `configs/eval_subjective_mtbench101_plus_api_small_part1.py`
- `configs/eval_subjective_mtbench101_plus_api_small_part2.py`
- `configs/eval_subjective_mtbench101_plus_api_small_part3.py`

同时新增运行脚本：

- `tools/run_mtbench101_plus_api_small_4way.py`

---

## 4. 工程修复阶段：文件定位与切片问题
### 4.1 eval 阶段文件名定位异常
在实际运行中发现 SubjectiveEval 在读取预测文件与结果文件时，对带下划线数据集缩写的兼容性不足，导致结果文件无法被正确识别。

### 4.2 修复结果文件定位逻辑
针对这一问题，修改了 SubjectiveEval 链路中的文件定位方式，使其能够正确处理：

- 带下划线数据集 `abbr`
- 分片文件名
- partial result 文件

### 4.3 修复重复切片问题
在 part3 等分片上进一步发现：对于已经经过 `test_range` 切片的数据集，eval 阶段仍再次切片，导致：

- 空预测列表；
- 结果错位；
- 分片评测失败。

对此修复了 SubjectiveEval 对已切片数据集的重复切片问题。

---

## 5. 关键问题定位：meta prompt 冲突
### 5.1 现象
在新的 MT-Bench-101+ 评测中，meta-judge 阶段出现大量样本无法提取分数，表现为：

- 空输出；
- 格式漂移；
- 结果目录虽生成，但后处理提取率较低。

### 5.2 定位方法
通过抽样检查 meta 结果文件与 prompt 结构，发现 meta-judge 的输出格式要求在 system prompt 和 user prompt 层面存在冲突。

### 5.3 根因
meta-judge 的 system prompt 仍继承了普通 judge 的逻辑，导致：

- 普通 judge 风格要求 `Rating: [[score]]`
- meta-judge user prompt 要求 `Final Rating: [[score]]`

模型在双重冲突指令下出现不稳定输出。

### 5.4 修复
在 `opencompass/datasets/subjective/mtbench101.py` 中重写 `meta_prompt_construct()`，为 meta-judge 单独设计 system prompt，明确其职责：

- 综合多评委意见；
- 不机械平均；
- 输出最终公平评分；
- 以 `Final Rating: [[score]]` 结尾。

### 5.5 验证效果
修复前：
- 成功提取：`68 / 100`

修复后：
- 成功提取：`92 / 100`

说明 prompt 冲突是提取率下降的主要根因。

---

## 6. 剩余失败样本分析
在局部验证后，对剩余失败样本进一步分析，发现：

- 剩余样本全部为空输出；
- 主要集中在 `GR` 任务；
- prompt 较长。

这说明系统性 prompt 冲突已被解决，剩余问题主要属于长 prompt 场景下的边缘不稳定样本。

---

## 7. 模型尝试与策略调整
### 7.1 尝试 glm-5.1
曾尝试将评测模型切换到 `glm-5.1`，但在 infer 阶段连续遇到 API 网络错误：

- `code=1234`
- 多次重试后仍失败

因此该模型未作为最终评测方案。

### 7.2 回退并重新选择 judge/meta 模型
随后回到更稳定的 GLM 系列，并继续尝试其它 judge / meta-judge 配置。

### 7.3 发现 API 额度问题
在 4 分片全流程并发实验时，infer 大体完成，但 eval 阶段出现：

- `code=1113`
- 余额不足或无可用资源包

这说明：

- infer 结果可以复用；
- 继续重跑 all 没有必要；
- 应优先补跑 eval。

---

## 8. 最终稳定方案形成
### 8.1 清理失败 eval 结果
在确认 infer 已完成且可复用后，删除了失败运行留下的 eval 结果目录，避免半截结果污染后续补跑。

### 8.2 切换到 glm-4.6v
最终将 4 个分片配置中的 persona judges 与 meta-judge 模型切换为 `glm-4.6v`。

### 8.3 仅补跑 eval
通过 `--mode eval --reuse <timestamp>` 的方式，复用既有 predictions，仅重跑主观评测与汇总阶段。

这一策略显著降低了成本，并成功完成后续实验。

---

## 9. 最终实验结果
### 9.1 分片结果
四个分片最终提取情况如下：

- part0: `100 / 100`
- part1: `98 / 100`
- part2: `99 / 100`
- part3: `100 / 100`

### 9.2 汇总结果
总计：

- 样本数：`400`
- 成功提取：`397`
- 提取率：`99.25%`

task-level 最终汇总分数：

- `GR = 8.9859`
- `IC = 8.2206`

### 9.3 结论
实验表明，MT-Bench-101+ 改造后的主观评测框架已经具备：

- 可运行性；
- 可补跑性；
- 可汇总性；
- 高提取率；
- 多视角评估能力。

---

## 10. 项目收尾与迁移准备
在改造阶段结束后，又完成了以下收尾工作：

1. 删除测试阶段输出结果文件；
2. 导出 Conda 环境到 `environment_opencompass310.yml`；
3. 清理无关中间文件；
4. 将项目推送到个人 Git 仓库；
5. 将分支统一为 `main`；
6. 开始整理论文图表、伪代码与日志材料。

---

## 11. 当前阶段定位
截至目前，本项目已完成“改造与跑通”阶段，进入“实验整理与论文写作”阶段。后续重点将转向：

- 论文图表与伪代码整理；
- 方法章节撰写；
- 结果与案例分析撰写；
- 新电脑环境迁移与后续实验扩展。
