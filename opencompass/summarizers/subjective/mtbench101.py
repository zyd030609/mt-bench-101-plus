"""MT-Bench-101 结果汇总器。"""

# flake8: noqa: E501
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime

from mmengine import ConfigDict

try:
    from prettytable import from_csv
except ImportError:
    from_csv = None

from opencompass.utils import model_abbr_from_cfg

from .compass_arena import CompassArenaSummarizer
from .utils import get_judgeanswer_and_reference, get_outdir


def post_process_mtbench_pair(judgement: str):
    pattern = r'\[([A-C]+)\]'
    matched_result = re.findall(pattern, judgement)
    if matched_result:
        return matched_result[0]
    return None


def post_process_mtbench101(judgement: str):
    match = re.search(r'(?:Final Rating|Rating):\s*\[\[?([0-9]+)\]?\]', judgement)
    if not match:
        match = re.search(r'\[([0-9]+)\]', judgement)
    if not match:
        return None
    score = int(match.group(1))
    return {'score': score, 'judgement': judgement}


def get_final_results(judged_answers, references, output_dir, fout_flag, model):
    task_multi_id_scores = defaultdict(list)
    task_scores = defaultdict(list)
    for ans, ref in zip(judged_answers, references):
        task = ref['task']
        multi_id = ref['multi_id']
        score = ans['score']
        task_multi_id_scores[(task, multi_id)].append(score)
    for (task, multi_id), scores in task_multi_id_scores.items():
        min_score = min(scores)
        task_scores[task].append(min_score)
    final_task_scores = {
        task: sum(scores) / len(scores) if scores else 0
        for task, scores in task_scores.items()
    }
    fout = osp.join(output_dir, 'task_score.csv')
    columns = list(final_task_scores.keys())
    print('================task_score=====================')
    print(final_task_scores)
    with open(fout, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0:
            writer.writerow(['model'] + columns)
        writer.writerow([model] + [final_task_scores[column] for column in columns])
    return 0


class MTBench101Summarizer(CompassArenaSummarizer):

    def __init__(self, config: ConfigDict, judge_type='single') -> None:
        self.tasks = []
        self.cfg = config
        self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
        self.eval_model_abbrs = [model_abbr_from_cfg(model) for model in self.eval_model_cfgs]
        self.judge_abbr = None
        if self.cfg.get('meta_judge_model', None) is not None:
            self.judge_abbr = model_abbr_from_cfg(self.cfg['meta_judge_model'])
        elif self.cfg.get('judge_models', None):
            self.judge_abbr = model_abbr_from_cfg(self.cfg['judge_models'][0])
        self.judge_function = post_process_mtbench101

    def _get_result_subdir(self, eval_model_abbr: str) -> str:
        if self.cfg.get('meta_judge_model', None) is not None:
            return eval_model_abbr + '_summarized-by--' + self.judge_abbr
        return eval_model_abbr + '_judged-by--' + self.judge_abbr

    def summarize(self, time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        dataset_cfgs = self.cfg['datasets']
        output_dir, results_folder = get_outdir(self.cfg, time_str)
        fout_flag = 0
        for eval_model_abbr in self.eval_model_abbrs:
            subdir = self._get_result_subdir(eval_model_abbr)
            subdir_path = os.path.join(results_folder, subdir)
            if os.path.isdir(subdir_path):
                model = eval_model_abbr
                for dataset in dataset_cfgs:
                    print()
                    judged_answers, references = get_judgeanswer_and_reference(
                        dataset, subdir_path, self.judge_function)
                    get_final_results(judged_answers, references, output_dir, fout_flag, model)
                    fout_flag += 1
            else:
                print(subdir_path + ' is not exist! please check!')
