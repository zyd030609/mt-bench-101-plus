"""MT-Bench-101 的数据集配置。

本文件只做“配置拼装”，不做复杂逻辑实现：
- 指定读哪些字段 (`reader_cfg`)
- 指定被评模型如何回答 (`infer_cfg`)
- 指定评委模型如何打分 (`eval_cfg`)
- 最终构造出 `subjective_datasets` 给顶层配置使用

真正的数据清洗、prompt 构造和样本展开逻辑在
`opencompass/datasets/subjective/mtbench101.py`。
"""

# PromptTemplate：负责把 `{变量}` 替换成真实内容。
from opencompass.openicl.icl_prompt_template import PromptTemplate
# ZeroRetriever：不做检索，直接使用当前样本本身。
from opencompass.openicl.icl_retriever import ZeroRetriever
# ChatInferencer：用于让被评模型在多轮对话里继续作答。
# GenInferencer 虽然这里导入了，但当前文件里没有实际使用。
from opencompass.openicl.icl_inferencer import ChatInferencer, GenInferencer
# LMEvaluator：让评委模型根据 prompt 生成主观评分。
from opencompass.openicl.icl_evaluator import LMEvaluator
# MTBench101Dataset：真正的数据集实现类。
from opencompass.datasets import MTBench101Dataset


# `reader_cfg` 定义数据集样本中有哪些字段会被 OpenCompass 读入。
# 这里的字段来自 `MTBench101Dataset.load()` 返回的样本字典。
subjective_reader_cfg = dict(
    # `input_columns` 表示运行时会读入这些字段。
    input_columns=[
        # `dialogue`：给被评模型看的多轮对话上下文。
        'dialogue',
        # `task`：当前样本对应的任务标签，如 CM / GR。
        'task',
        # `multi_id`：整段多轮对话的编号。
        'multi_id',
        # `turn_id`：当前是这段对话里的第几轮评分样本。
        'turn_id',
        # `system_prompt`：给评委看的系统级评分说明。
        'system_prompt',
        # `prompt_template`：给评委看的用户输入模板。
        'prompt_template',
    ],
    # `output_column='judge'` 表示参考输出字段叫 `judge`。
    # 这个字段不是真正的文本答案，而是用于对齐与汇总的元信息。
    output_column='judge',
)

# 这里列出要构造哪些 MT-Bench-101 数据集变体。
# 当前只有一个标准集：`mtbench101`。
subjective_all_sets = [
    'mtbench101',
]

# 数据文件所在目录。
data_path = 'data/subjective/'

# 最终导出的数据集配置列表。
subjective_datasets = []

# 对列表中的每个数据集名字，分别构造一份完整配置。
for _name in subjective_all_sets:
    # -----------------------------
    # 1) 被评模型回答阶段的配置
    # -----------------------------
    subjective_infer_cfg = dict(
        # `prompt_template` 决定如何把数据集字段喂给被评模型。
        prompt_template=dict(
            type=PromptTemplate,
            # 这里直接把 `dialogue` 字段作为模板。
            # 因为 `dialogue` 本身已经是多轮消息结构，不需要再额外拼复杂文本。
            template="""{dialogue}""",
        ),
        # `ZeroRetriever` 表示不做额外示例检索。
        retriever=dict(type=ZeroRetriever),
        # `ChatInferencer` 让模型在聊天场景里继续生成最后一轮回答。
        inferencer=dict(
            type=ChatInferencer,
            # judge / 被评模型上下文上限。
            max_seq_len=4096,
            # 最多生成多少 token。
            max_out_len=4096,
            # `infer_mode='last'` 表示只补全最后一轮 BOT 回复。
            infer_mode='last',
        ),
    )

    # -----------------------------
    # 2) 评委模型打分阶段的配置
    # -----------------------------
    subjective_eval_cfg = dict(
        evaluator=dict(
            # 使用语言模型评委器。
            type=LMEvaluator,
            # 评委模型同样需要一个 PromptTemplate。
            prompt_template=dict(
                type=PromptTemplate,
                # 这里不是普通字符串，而是“带角色的对话模板”。
                template=dict(
                    # `begin` 段放系统提示词。
                    begin=[
                        dict(
                            # 把这一段标记成 SYSTEM 角色。
                            role='SYSTEM',
                            # 若目标模型不支持 SYSTEM，可退化成 HUMAN。
                            fallback_role='HUMAN',
                            # `{system_prompt}` 会被替换成评分规则说明。
                            prompt='{system_prompt}')
                    ],
                    # `round` 段放用户输入，也就是待评分内容。
                    round=[
                        dict(
                            role='HUMAN',
                            # `{prompt_template}` 里会包含对话历史和 `{prediction}` 占位符。
                            prompt='{prompt_template}'
                        ),
                    ]),
            ),
        ),
        # `pred_role='BOT'` 表示后处理时，只取模型输出里的 BOT 回复部分。
        pred_role='BOT',
    )

    # 把当前数据集配置追加到列表中。
    subjective_datasets.append(
        dict(
            # 数据集缩写。
            abbr=f'{_name}',
            # 指定数据集实现类。
            type=MTBench101Dataset,
            # 数据目录。
            path=data_path,
            # 数据文件名的一部分；最终会去读 `data/subjective/mtbench101.jsonl`。
            name=_name,
            # 指定读取哪些字段。
            reader_cfg=subjective_reader_cfg,
            # 指定被评模型回答阶段如何跑。
            infer_cfg=subjective_infer_cfg,
            # 指定评委模型评分阶段如何跑。
            eval_cfg=subjective_eval_cfg
        ))
