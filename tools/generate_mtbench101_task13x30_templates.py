import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / 'configs' / 'mtbench101_plus_task13x30'
CFG_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ['AR', 'CC', 'CM', 'CR', 'FR', 'GR', 'IC', 'MR', 'PI', 'SA', 'SC', 'SI', 'TS']
JUDGE_TASK_KEYS = {
    'AR': 'sk-95fba73d54aa427a8b4ffa681085afba',
    'CC': 'sk-70de4267902041e9b2c8700a3a28fe05',
    'CM': 'sk-b028db3057634fbba8eee585dfafbdf7',
    'CR': 'sk-ce0090bda7e445949bb85812b7f0e2be',
    'FR': 'sk-ce3458b5c1334d9bb2dc7a08f0d54864',
    'GR': 'sk-22de5d5bd52b40c885f6956b1098a09b',
    'IC': 'sk-c080469936ce4debbc65d547456859e8',
    'MR': 'sk-a0deca8e95ba428f9481457918996435',
    'PI': 'sk-e29b9b3766bd4bddb7d84882874be004',
    'SA': 'sk-33c64d7275a040f18cc83560069d0a7a',
    'SC': 'sk-3d4e05222bea4906b1b60ceb3d46d5aa',
    'SI': 'sk-1274681d784a436c8de52b00e5096ea3',
    'TS': 'sk-8530bb90b8894712a8689ca213d1db99',
}
TARGET_TASK_KEYS = {
    'AR': 'b7d7f752b9ba4fd2b8820907bd5e3227.wgGksj2q6P2YOe1f',
    'CC': 'eb47f173e4e54e5a8af2f0b41a47033b.4OF7U9pIALKnBTuU',
    'CM': '0f05be18202d416f8ce7c991f4818eaf.5ElL6Ztxmri2EW4E',
    'CR': 'e5bd37e60c60439797056fa8459217ca.6x1barxc91k7f2M4',
    'FR': '1bb43f97051a46f3b4572b9e154702ba.DAjCC4qTM0LvkoNe',
    'GR': 'd80ec00c16ef42f0a4db1392aeb6d54d.nrP9N5fNgPyFeuMc',
    'IC': '6fb6c0e34dc14af2985195cbfd7ded4e.RujBTiRxFpSaQyrI',
    'MR': 'f6f0a2938d5e46789619931aef35d598.cB4pCUOOu9KIFMsH',
    'PI': 'd4887a62838947f096ace4eb5a7ef4e0.9pswrHikS1dJaArV',
    'SA': '5094d62af5a24c49a74117cd8b952a87.hvPvcT48bWi77JEH',
    'SC': '33a4b6d8b2e54d5b8cb7f53009a106b4.PXPfABpBaCre6UTI',
    'SI': '3634654acaca463996e6e6374ee7e62b.uOGy17xwqqlSyL0x',
    'TS': '8aed156593ee441db03fa6fd609c38ba.tilkBoGdh2TZumaE',
}

TARGET_MODEL_PATH = 'glm-4.6v-flashx'
TARGET_MODEL_ABBR = 'glm-4_6v-flashx'
TARGET_API_BASE = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
JUDGE_MODEL_PATH = 'deepseek-v3.2'
JUDGE_API_BASE = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'

# 关闭深度思考：
# 1) 智谱 OpenAI 兼容接口使用 thinking.type = disabled
# 2) DashScope 兼容接口对深度思考模型常见写法为 enable_thinking = False
TARGET_EXTRA_BODY = {'thinking': {'type': 'disabled'}}
JUDGE_EXTRA_BODY = {'enable_thinking': False}

TEMPLATE = '''from mmengine.config import read_base\n\nwith read_base():\n    from ..datasets.subjective.multiround.mtbench101_plus_judge import subjective_datasets_plus_meta\n\nfrom opencompass.models import OpenAI\nfrom opencompass.partitioners import NaivePartitioner\nfrom opencompass.partitioners.sub_size import SubjectiveSizePartitioner\nfrom opencompass.runners import LocalRunner\nfrom opencompass.runners.local_api import LocalAPIRunner\nfrom opencompass.summarizers import MTBench101Summarizer\nfrom opencompass.tasks import OpenICLInferTask\nfrom opencompass.tasks.subjective_eval import SubjectiveEvalTask\n\napi_meta_template = dict(\n    round=[\n        dict(role='SYSTEM', api_role='SYSTEM'),\n        dict(role='HUMAN', api_role='HUMAN'),\n        dict(role='BOT', api_role='BOT', generate=True),\n    ])\n\npersona_output_format = (\n    'Positive Evidence: <one concise paragraph>\\n'\n    'Negative Evidence: <one concise paragraph>\\n'\n    'User-perspective Concern: <one concise paragraph>\\n'\n    'Rating: [[score]]'\n)\n\nfinal_output_format = (\n    'Consensus: <one concise paragraph>\\n'\n    'Main Disagreement: <one concise paragraph>\\n'\n    'Final Justification: <one concise paragraph>\\n'\n    'Final Rating: [[score]]'\n)\n\nJUDGE_TASK_KEY = {judge_task_key!r}\nTARGET_TASK_KEY = {target_task_key!r}\nTARGET_MODEL_PATH = {target_model_path!r}\nTARGET_MODEL_ABBR = {target_model_abbr!r}\nTARGET_EXTRA_BODY = {target_extra_body!r}\nJUDGE_EXTRA_BODY = {judge_extra_body!r}\n\nmodels = [\n    dict(\n        abbr=TARGET_MODEL_ABBR,\n        type=OpenAI,\n        path=TARGET_MODEL_PATH,\n        key=TARGET_TASK_KEY,\n        openai_api_base={target_api_base!r},\n        extra_body=TARGET_EXTRA_BODY,\n        meta_template=api_meta_template,\n        query_per_second=1,\n        max_out_len=4096,\n        max_seq_len=4096,\n        batch_size=1,\n        temperature=0.0,\n    ),\n]\n\njudge_models = [\n    dict(\n        abbr='judge-analytic-rigor',\n        type=OpenAI,\n        path={judge_model_path!r},\n        key=JUDGE_TASK_KEY,\n        openai_api_base={judge_api_base!r},\n        extra_body=JUDGE_EXTRA_BODY,\n        meta_template=api_meta_template,\n        query_per_second=1,\n        max_out_len=4096,\n        max_seq_len=4096,\n        batch_size=1,\n        temperature=0.0,\n        judge_panel_name='MT-Bench-101+ persona panel',\n        persona_name='Analytical Rigor Judge',\n        persona_description='A strict evaluator who prioritizes correctness, internal consistency, faithful grounding in the dialogue history, and precise reasoning.',\n        persona_focus='logical consistency, criterion coverage, contradiction detection, and faithfulness to the conversation context',\n        persona_output_format=persona_output_format,\n    ),\n    dict(\n        abbr='judge-practical-task',\n        type=OpenAI,\n        path={judge_model_path!r},\n        key=JUDGE_TASK_KEY,\n        openai_api_base={judge_api_base!r},\n        extra_body=JUDGE_EXTRA_BODY,\n        meta_template=api_meta_template,\n        query_per_second=1,\n        max_out_len=4096,\n        max_seq_len=4096,\n        batch_size=1,\n        temperature=0.0,\n        judge_panel_name='MT-Bench-101+ persona panel',\n        persona_name='Practical Task Judge',\n        persona_description='An evaluator focused on whether the answer is directly useful, actionable, complete enough to solve the user task, and aligned with real user needs.',\n        persona_focus='task completion, usefulness, specificity, actionability, and whether the response would help the user finish the task',\n        persona_output_format=persona_output_format,\n    ),\n    dict(\n        abbr='judge-communication-experience',\n        type=OpenAI,\n        path={judge_model_path!r},\n        key=JUDGE_TASK_KEY,\n        openai_api_base={judge_api_base!r},\n        extra_body=JUDGE_EXTRA_BODY,\n        meta_template=api_meta_template,\n        query_per_second=1,\n        max_out_len=4096,\n        max_seq_len=4096,\n        batch_size=1,\n        temperature=0.0,\n        judge_panel_name='MT-Bench-101+ persona panel',\n        persona_name='Communication Experience Judge',\n        persona_description='An evaluator focused on clarity, interaction quality, coherence, tone, smoothness, and whether the response feels helpful and trustworthy to a user.',\n        persona_focus='clarity, communicative quality, naturalness, structure, user comfort, and whether the response is easy to follow',\n        persona_output_format=persona_output_format,\n    ),\n]\n\nmeta_judge_model = dict(\n    abbr='meta-judge-synthesizer',\n    type=OpenAI,\n    path={judge_model_path!r},\n    key=JUDGE_TASK_KEY,\n    openai_api_base={judge_api_base!r},\n    extra_body=JUDGE_EXTRA_BODY,\n    meta_template=api_meta_template,\n    query_per_second=1,\n    max_out_len=4096,\n    max_seq_len=4096,\n    batch_size=1,\n    temperature=0.0,\n    judge_panel_name='MT-Bench-101+ persona panel',\n    final_output_format=final_output_format,\n)\n\ndatasets = [dict(\n    abbr='mtbench101_task13x30_{task}',\n    type=subjective_datasets_plus_meta[0]['type'],\n    path='data/subjective',\n    name='mtbench101_task13x30_by_task/mtbench101_task13x30_{task}',\n    prompt_mode='meta',\n    judge_count=3,\n    reader_cfg=subjective_datasets_plus_meta[0]['reader_cfg'],\n    infer_cfg=subjective_datasets_plus_meta[0]['infer_cfg'],\n    eval_cfg=subjective_datasets_plus_meta[0]['eval_cfg'],\n)]\n\ninfer = dict(\n    partitioner=dict(type=NaivePartitioner),\n    runner=dict(\n        type=LocalAPIRunner,\n        max_num_workers=1,\n        concurrent_users=1,\n        task=dict(type=OpenICLInferTask),\n    ),\n)\n\neval = dict(\n    partitioner=dict(\n        type=SubjectiveSizePartitioner,\n        max_task_size=100000,\n        mode='singlescore',\n        models=models,\n        judge_models=judge_models,\n        meta_judge_model=meta_judge_model,\n    ),\n    runner=dict(\n        type=LocalRunner,\n        max_num_workers=1,\n        task=dict(type=SubjectiveEvalTask),\n    ),\n)\n\nsummarizer = dict(type=MTBench101Summarizer, judge_type='single')\nwork_dir = 'outputs/mtbench101_plus_task13x30/{target_model_abbr}/{task}/'\n'''

manifest = {
    'tasks': TASKS,
    'judge_model_path': JUDGE_MODEL_PATH,
    'judge_api_base': JUDGE_API_BASE,
    'target_model_path': TARGET_MODEL_PATH,
    'target_api_base': TARGET_API_BASE,
    'target_extra_body': TARGET_EXTRA_BODY,
    'judge_extra_body': JUDGE_EXTRA_BODY,
}

for task in TASKS:
    cfg_path = CFG_DIR / f'eval_mtbench101_plus_task13x30_{task}.py'
    cfg_path.write_text(
        TEMPLATE.format(
            task=task,
            judge_task_key=JUDGE_TASK_KEYS[task],
            target_task_key=TARGET_TASK_KEYS[task],
            target_model_path=TARGET_MODEL_PATH,
            target_model_abbr=TARGET_MODEL_ABBR,
            target_api_base=TARGET_API_BASE,
            judge_model_path=JUDGE_MODEL_PATH,
            judge_api_base=JUDGE_API_BASE,
            target_extra_body=TARGET_EXTRA_BODY,
            judge_extra_body=JUDGE_EXTRA_BODY,
        ),
        encoding='utf-8')

(CFG_DIR / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(str(CFG_DIR))
