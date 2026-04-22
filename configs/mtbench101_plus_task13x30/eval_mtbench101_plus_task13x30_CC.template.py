from mmengine.config import read_base

with read_base():
    from ..datasets.subjective.multiround.mtbench101_plus_judge import subjective_datasets_plus_meta

from opencompass.models import OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.summarizers import MTBench101Summarizer
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

api_meta_template = dict(
    round=[
        dict(role='SYSTEM', api_role='SYSTEM'),
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ])

persona_output_format = (
    'Positive Evidence: <one concise paragraph>\n'
    'Negative Evidence: <one concise paragraph>\n'
    'User-perspective Concern: <one concise paragraph>\n'
    'Rating: [[score]]'
)

final_output_format = (
    'Consensus: <one concise paragraph>\n'
    'Main Disagreement: <one concise paragraph>\n'
    'Final Justification: <one concise paragraph>\n'
    'Final Rating: [[score]]'
)

JUDGE_TASK_KEY = 'ENV_CC_KEY'
TARGET_TASK_KEY = 'ENV_TARGET_CC_KEY'
TARGET_MODEL_PATH = '{target_model_path}'
TARGET_MODEL_ABBR = '{target_model_abbr}'

models = [
    dict(
        abbr=TARGET_MODEL_ABBR,
        type=OpenAI,
        path=TARGET_MODEL_PATH,
        key=TARGET_TASK_KEY,
        openai_api_base='{target_api_base}',
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=4096,
        max_seq_len=4096,
        batch_size=1,
        temperature=0.0,
    ),
]

judge_models = [
    dict(
        abbr='judge-analytic-rigor',
        type=OpenAI,
        path='{judge_model_path}',
        key=JUDGE_TASK_KEY,
        openai_api_base='{judge_api_base}',
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=4096,
        max_seq_len=4096,
        batch_size=1,
        temperature=0.0,
        judge_panel_name='MT-Bench-101+ persona panel',
        persona_name='Analytical Rigor Judge',
        persona_description='A strict evaluator who prioritizes correctness, internal consistency, faithful grounding in the dialogue history, and precise reasoning.',
        persona_focus='logical consistency, criterion coverage, contradiction detection, and faithfulness to the conversation context',
        persona_output_format=persona_output_format,
    ),
    dict(
        abbr='judge-practical-task',
        type=OpenAI,
        path='{judge_model_path}',
        key=JUDGE_TASK_KEY,
        openai_api_base='{judge_api_base}',
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=4096,
        max_seq_len=4096,
        batch_size=1,
        temperature=0.0,
        judge_panel_name='MT-Bench-101+ persona panel',
        persona_name='Practical Task Judge',
        persona_description='An evaluator focused on whether the answer is directly useful, actionable, complete enough to solve the user task, and aligned with real user needs.',
        persona_focus='task completion, usefulness, specificity, actionability, and whether the response would help the user finish the task',
        persona_output_format=persona_output_format,
    ),
    dict(
        abbr='judge-communication-experience',
        type=OpenAI,
        path='{judge_model_path}',
        key=JUDGE_TASK_KEY,
        openai_api_base='{judge_api_base}',
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=4096,
        max_seq_len=4096,
        batch_size=1,
        temperature=0.0,
        judge_panel_name='MT-Bench-101+ persona panel',
        persona_name='Communication Experience Judge',
        persona_description='An evaluator focused on clarity, interaction quality, coherence, tone, smoothness, and whether the response feels helpful and trustworthy to a user.',
        persona_focus='clarity, communicative quality, naturalness, structure, user comfort, and whether the response is easy to follow',
        persona_output_format=persona_output_format,
    ),
]

meta_judge_model = dict(
    abbr='meta-judge-synthesizer',
    type=OpenAI,
    path='{judge_model_path}',
    key=JUDGE_TASK_KEY,
    openai_api_base='{judge_api_base}',
    meta_template=api_meta_template,
    query_per_second=1,
    max_out_len=4096,
    max_seq_len=4096,
    batch_size=1,
    temperature=0.0,
    judge_panel_name='MT-Bench-101+ persona panel',
    final_output_format=final_output_format,
)

datasets = [dict(
    abbr='mtbench101_task13x30_CC',
    type=subjective_datasets_plus_meta[0]['type'],
    path='data/subjective',
    name='mtbench101_task13x30_by_task/mtbench101_task13x30_CC',
    prompt_mode='meta',
    judge_count=3,
    reader_cfg=subjective_datasets_plus_meta[0]['reader_cfg'],
    infer_cfg=subjective_datasets_plus_meta[0]['infer_cfg'],
    eval_cfg=subjective_datasets_plus_meta[0]['eval_cfg'],
)]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalAPIRunner,
        max_num_workers=1,
        concurrent_users=1,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner,
        max_task_size=100000,
        mode='singlescore',
        models=models,
        judge_models=judge_models,
        meta_judge_model=meta_judge_model,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=SubjectiveEvalTask),
    ),
)

summarizer = dict(type=MTBench101Summarizer, judge_type='single')
work_dir = 'outputs/mtbench101_plus_task13x30/{target_model_abbr}/CC/'
