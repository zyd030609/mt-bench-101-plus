from copy import deepcopy
from mmengine.config import read_base

with read_base():
    from .datasets.subjective.multiround.mtbench101_plus_judge import subjective_datasets_plus_meta

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

PART_KEY = '647424aa76944a148b65644d8079837d.K2iEibHejSydyIjh'

models = [
    dict(
        abbr='baseline-api-model',
        type=OpenAI,
        path='glm-4.6v',
        key=PART_KEY,
        openai_api_base='https://open.bigmodel.cn/api/paas/v4/chat/completions',
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
        path='glm-4.6v',
        key=PART_KEY,
        openai_api_base='https://open.bigmodel.cn/api/paas/v4/chat/completions',
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
        path='glm-4.6v',
        key=PART_KEY,
        openai_api_base='https://open.bigmodel.cn/api/paas/v4/chat/completions',
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
        path='glm-4.6v',
        key=PART_KEY,
        openai_api_base='https://open.bigmodel.cn/api/paas/v4/chat/completions',
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
    path='glm-4.6v',
    key=PART_KEY,
    openai_api_base='https://open.bigmodel.cn/api/paas/v4/chat/completions',
    meta_template=api_meta_template,
    query_per_second=1,
    max_out_len=4096,
    max_seq_len=4096,
    batch_size=1,
    temperature=0.0,
    judge_panel_name='MT-Bench-101+ persona panel',
    final_output_format=final_output_format,
)

datasets = deepcopy(subjective_datasets_plus_meta)
datasets[0]['reader_cfg']['test_range'] = '[0:100]'
datasets[0]['abbr'] = 'mtbench101_plus_slice_0_100'

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
work_dir = 'outputs/mtbench101_plus_api_small_4way/part0/'
