"""MT-Bench-101+ 数据集配置。"""

from opencompass.datasets import MTBench101Dataset
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

subjective_reader_cfg = dict(
    input_columns=[
        'dialogue',
        'task',
        'multi_id',
        'turn_id',
        'system_prompt',
        'prompt_template',
    ],
    output_column='judge',
)

subjective_all_sets = ['mtbench101']
data_path = 'data/subjective/'
subjective_datasets = []
subjective_datasets_plus_meta = []

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

for _name in subjective_all_sets:
    infer_cfg = dict(
        prompt_template=dict(type=PromptTemplate, template="""{dialogue}"""),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=ChatInferencer,
            max_seq_len=4096,
            max_out_len=4096,
            infer_mode='last',
        ),
    )

    eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt='{system_prompt}')
                    ],
                    round=[dict(role='HUMAN', prompt='{prompt_template}')],
                ),
            ),
        ),
        pred_role='BOT',
    )

    meta_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt='{system_prompt}')
                    ],
                    round=[dict(role='HUMAN', prompt='{prompt_template}')],
                ),
            ),
            meta_review_prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt='{system_prompt}')
                    ],
                    round=[dict(role='HUMAN', prompt='{prompt_template}')],
                ),
            ),
        ),
        pred_role='BOT',
    )

    subjective_datasets.append(
        dict(
            abbr=f'{_name}',
            type=MTBench101Dataset,
            path=data_path,
            name=_name,
            prompt_mode='persona',
            reader_cfg=subjective_reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg,
        ))

    subjective_datasets_plus_meta.append(
        dict(
            abbr=f'{_name}',
            type=MTBench101Dataset,
            path=data_path,
            name=_name,
            prompt_mode='meta',
            judge_count=3,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=meta_eval_cfg,
        ))
