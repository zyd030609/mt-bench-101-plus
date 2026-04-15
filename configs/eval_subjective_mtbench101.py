"""MT-Bench-101 主观评测总配置。

这个文件是跑通 MT-Bench-101 的顶层入口，负责把整个实验链路串起来：
1. 指定被评模型 `models`
2. 指定数据集配置 `datasets`
3. 配置回答生成阶段 `infer`
4. 配置评委打分阶段 `eval`
5. 配置结果汇总 `summarizer`

阅读顺序建议：先看本文件，再看 `configs/datasets/.../mtbench101_judge.py`，
随后看 `opencompass/datasets/subjective/mtbench101.py` 与
`opencompass/tasks/subjective_eval.py`。
"""

# `read_base` 是 mmengine 的配置继承工具。
# 它允许当前配置文件去“导入”别的配置文件里的变量。
from mmengine.config import read_base

# 这里进入配置继承上下文。
# 下面这行不是普通 Python import，而是 mmengine 风格的配置拼装写法。
with read_base():
    # 从数据集配置文件里导入 `subjective_datasets`。
    # 这个变量稍后会直接赋值给顶层的 `datasets`。
    from .datasets.subjective.multiround.mtbench101_judge import subjective_datasets

# 导入当前配置真正会用到的模型类：
# - `HuggingFaceChatGLM3`：被评模型
# - `OpenAI`：评委模型
from opencompass.models import HuggingFaceChatGLM3, OpenAI

# 导入当前配置真正会用到的任务切分器。
# `SizePartitioner` 用于推理阶段；`SubjectiveSizePartitioner` 用于主观评审阶段。
from opencompass.partitioners import SizePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner

# 导入任务运行器。
# `LocalRunner` 在本地直接执行任务。
# `SlurmSequentialRunner` 用于有 Slurm 集群的环境。
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner

# 导入两个核心任务：
# - `OpenICLInferTask`：让被评模型先回答问题
# - `SubjectiveEvalTask`：让评委模型读取回答并打分
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

# 导入 MT-Bench-101 专用汇总器。
from opencompass.summarizers import MTBench101Summarizer

# ---------------------------------------------------------------------------------------------------------

# `api_meta_template` 描述多轮对话时的角色结构。
# 对很多 chat 模型 / API 模型来说，输入并不是一整段纯文本，
# 而是带有角色信息的消息列表：SYSTEM / HUMAN / BOT。
api_meta_template = dict(
    round=[
        # SYSTEM：系统提示词，通常用来规定助手身份或规则。
        dict(role='SYSTEM', api_role='SYSTEM'),
        # HUMAN：用户输入。
        dict(role='HUMAN', api_role='HUMAN'),
        # BOT：模型回复；`generate=True` 表示这一轮由模型来生成。
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

# -------------Inference Stage ----------------------------------------

# `models` 存放“被评模型”的配置，而不是评委模型。
# 这些模型会先阅读 MT-Bench-101 的多轮对话上下文，并生成自己的回答。
# 后续评委模型打分时，看的就是这里生成出来的回答。
models = [
    dict(
        # 指定模型类型：这里用 ChatGLM3 的 HuggingFace 封装类。
        type=HuggingFaceChatGLM3,
        # `abbr` 是缩写，会直接影响输出目录名。
        abbr='chatglm3-6b-hf',
        # `path` 是模型权重路径或 HuggingFace 仓库名。
        path='THUDM/chatglm3-6b',
        # `tokenizer_path` 是 tokenizer 的路径；通常与模型路径相同。
        tokenizer_path='THUDM/chatglm3-6b',
        # `model_kwargs` 是构造底层模型时要传入的参数。
        model_kwargs=dict(
            # `device_map='auto'` 让 transformers 自动分配设备。
            device_map='auto',
            # 允许加载需要自定义代码的模型仓库。
            trust_remote_code=True,
        ),
        # `tokenizer_kwargs` 是构造 tokenizer 的参数。
        tokenizer_kwargs=dict(
            # padding 放在左边，适合很多自回归模型。
            padding_side='left',
            # 截断也从左边开始。
            truncation_side='left',
            # 同样允许远程自定义代码。
            trust_remote_code=True,
        ),
        # `generation_kwargs` 是生成回答时的参数。
        generation_kwargs=dict(
            # `do_sample=True` 表示采样生成；
            # 主观评测常常允许模型有更自由的表达，不强制贪心解码。
            do_sample=True,
        ),
        # 指定该模型使用的多轮对话角色模板。
        meta_template=api_meta_template,
        # 最多生成多少 token。
        max_out_len=4096,
        # 模型最多接收多少 token 的上下文。
        max_seq_len=4096,
        # 一次推理的 batch 大小。
        batch_size=1,
        # 运行配置：这里表示至少需要 2 张 GPU，单进程执行。
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]

# 顶层数据集直接复用前面导入的 `subjective_datasets`。
# `*` 的意思是把列表展开，再组成一个新的列表。
datasets = [*subjective_datasets]

# `infer` 定义“回答生成阶段”的执行方式。
# 这一阶段的目标不是打分，而是让被评模型先回答问题。
infer = dict(
    # `partitioner` 负责把大任务拆成小任务。
    partitioner=dict(
        # 按数据规模切分。
        type=SizePartitioner,
        # 每个子任务最大大小。
        max_task_size=10000,
    ),
    # `runner` 负责实际执行这些子任务。
    runner=dict(
        # 当前配置默认使用 Slurm 顺序运行器。
        type=SlurmSequentialRunner,
        # Slurm 分区名。
        partition='llm_dev2',
        # 配额类型。
        quotatype='auto',
        # 最多并行多少个 worker。
        max_num_workers=32,
        # 每个子任务实际执行的任务类型。
        task=dict(type=OpenICLInferTask),
    ),
)

# -------------Evalation Stage ----------------------------------------

# `judge_models` 存放“评委模型”的配置。
# 评委模型不会参与回答问题，而是负责阅读被评模型的回答并给出评分。
judge_models = [dict(
    # 输出目录中使用的评委名称缩写。
    abbr='GPT4-Turbo',
    # 评委模型类型，这里是 OpenAI API 封装。
    type=OpenAI,
    # 指定具体使用的 OpenAI 模型。
    # 官方 README 里建议用这个版本来对齐 leaderboard。
    path='gpt-4-1106-preview',
    # API key；留空时会从环境变量 `OPENAI_API_KEY` 中读取。
    key='',
    # 同样采用前面定义的多轮角色模板。
    meta_template=api_meta_template,
    # 每秒允许发多少个请求。
    query_per_second=16,
    # judge 最多生成多少 token。
    max_out_len=4096,
    # judge 接收的最大上下文长度。
    max_seq_len=4096,
    # judge 推理 batch size。
    batch_size=8,
    # 评委温度；非 0 表示评分理由有一定随机性。
    temperature=0.8,
)]

# `eval` 定义“评委打分阶段”的执行方式。
# 这一阶段会读取前面 infer 产生的 predictions，再调用 judge model 进行主观评分。
eval = dict(
    partitioner=dict(
        # 主观评测用专门的分片器。
        type=SubjectiveSizePartitioner,
        # 主观任务分片上限。
        max_task_size=100000,
        # `singlescore` 表示单模型被单独打分，而不是模型两两对比。
        mode='singlescore',
        # 告诉分片器哪些是被评模型。
        models=models,
        # 告诉分片器有哪些评委模型。
        judge_models=judge_models,
    ),
    runner=dict(
        # 主观评测阶段默认在本地跑。
        type=LocalRunner,
        # 最大 worker 数。
        max_num_workers=32,
        # 具体执行 `SubjectiveEvalTask`。
        task=dict(type=SubjectiveEvalTask),
    ),
)

# `summarizer` 负责把 judge 的自然语言输出解析为最终分数表。
summarizer = dict(
    # 指定使用 MT-Bench-101 专用汇总器。
    type=MTBench101Summarizer,
    # 当前是单评委模式。
    judge_type='single',
)

# `work_dir` 是输出目录根路径。
# 后续会在这里生成 predictions / results / summary 等子目录。
work_dir = 'outputs/mtbench101/'
