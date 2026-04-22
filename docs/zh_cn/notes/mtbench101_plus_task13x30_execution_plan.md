# MT-Bench-101+ 子数据集实验执行方案（13任务 × 每任务30条 × 固化配置版）

## 1. 当前配置状态
本项目已改为“可直接运行”的固化配置版，以下信息已经直接写入项目文件：

- 待测模型：`glm-4.6v-flashx`
- 待测模型 API：`https://open.bigmodel.cn/api/paas/v4/chat/completions`
- 评委模型：`deepseek-v3.2`
- 评委模型 API：`https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`
- 13 个评委任务 key：已写入配置生成脚本
- 13 个待测任务 key：已写入配置生成脚本

## 2. 深度思考配置
按当前要求，待测模型和评委模型都关闭深度思考。

当前采用的接口层配置为：

- 待测模型（智谱兼容接口）：`extra_body = {"thinking": {"type": "disabled"}}`
- 评委模型（DashScope 兼容接口）：`extra_body = {"enable_thinking": False}`

同时，实验 prompt 仍维持评分导向，不额外引导输出思维链。

## 3. 关键文件
### 3.1 固化 key 与模型信息的配置生成脚本
- `tools/generate_mtbench101_task13x30_templates.py`

### 3.2 固化后的 13 个任务配置文件
目录：
- `configs/mtbench101_plus_task13x30/`

文件名：
- `eval_mtbench101_plus_task13x30_AR.py`
- `eval_mtbench101_plus_task13x30_CC.py`
- `eval_mtbench101_plus_task13x30_CM.py`
- `eval_mtbench101_plus_task13x30_CR.py`
- `eval_mtbench101_plus_task13x30_FR.py`
- `eval_mtbench101_plus_task13x30_GR.py`
- `eval_mtbench101_plus_task13x30_IC.py`
- `eval_mtbench101_plus_task13x30_MR.py`
- `eval_mtbench101_plus_task13x30_PI.py`
- `eval_mtbench101_plus_task13x30_SA.py`
- `eval_mtbench101_plus_task13x30_SC.py`
- `eval_mtbench101_plus_task13x30_SI.py`
- `eval_mtbench101_plus_task13x30_TS.py`

### 3.3 一键并行运行脚本
- `tools/run_mtbench101_task13x30_single_model.py`

## 4. 运行方式
### 4.1 首次完整运行
在项目根目录执行：

```powershell
& "D:\Anaconda\envs\opencompass310\python.exe" "tools/run_mtbench101_task13x30_single_model.py" --mode all --max-parallel 13
```

如果想更稳妥地先测试，可把并行度调低：

```powershell
& "D:\Anaconda\envs\opencompass310\python.exe" "tools/run_mtbench101_task13x30_single_model.py" --mode all --max-parallel 3
```

### 4.2 仅补跑评测阶段
如果 infer 已完成，只补跑 eval：

```powershell
& "D:\Anaconda\envs\opencompass310\python.exe" "tools/run_mtbench101_task13x30_single_model.py" --mode eval --max-parallel 13 --reuse <timestamp>
```

## 5. 结果输出位置
### 5.1 各任务输出目录
- `outputs/mtbench101_plus_task13x30/glm-4_6v-flashx/<task>/`

### 5.2 单轮总状态汇总
- `outputs/mtbench101_plus_task13x30/_run_summaries/`

其中包含：
- `run_args.json`
- `run_status.csv`

## 6. 迁移到另一台电脑时需要带走的文件
建议至少保留：

- `data/subjective/mtbench101_task13x30.jsonl`
- `data/subjective/mtbench101_task13x30_meta.json`
- `data/subjective/mtbench101_task13x30_by_task/`
- `configs/mtbench101_plus_task13x30/`
- `tools/build_mtbench101_task13x30.py`
- `tools/generate_mtbench101_task13x30_templates.py`
- `tools/run_mtbench101_task13x30_single_model.py`
- `opencompass/models/openai_api.py`
- 本文档 `docs/zh_cn/notes/mtbench101_plus_task13x30_execution_plan.md`

## 7. 注意事项
1. 当前 key 以明文形式写入项目文件，仅适合本地实验与迁移使用。
2. 如需提交仓库或分享给他人，务必先脱敏。
3. 如果后续需要更换待测模型或 key，只需修改 `tools/generate_mtbench101_task13x30_templates.py`，然后重新生成配置文件。
