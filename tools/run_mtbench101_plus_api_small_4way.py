import argparse
import csv
import os
import os.path as osp
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from mmengine import Config

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
PART_CONFIGS = [
    'configs/eval_subjective_mtbench101_plus_api_small_part0.py',
    'configs/eval_subjective_mtbench101_plus_api_small_part1.py',
    'configs/eval_subjective_mtbench101_plus_api_small_part2.py',
    'configs/eval_subjective_mtbench101_plus_api_small_part3.py',
]
SUMMARY_ROOT = osp.join(ROOT_DIR, 'outputs', 'mtbench101_plus_api_small_4way', 'merged_summary')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run four MTBench101+ GLM shards concurrently in one terminal and aggregate results.'
    )
    parser.add_argument('--python', default=sys.executable, help='Python executable to use.')
    parser.add_argument('--reuse', default=None, help='Reuse a specific timestamp for all shards.')
    parser.add_argument('--mode', default='all', choices=['all', 'infer', 'eval', 'viz'])
    parser.add_argument('--max-parallel', type=int, default=4, help='Maximum concurrent shard processes.')
    parser.add_argument('--stop-on-failure', action='store_true', help='Stop early after first failed shard is observed.')
    return parser.parse_args()


def load_cfg(config_rel_path: str):
    return Config.fromfile(osp.join(ROOT_DIR, config_rel_path))


def list_timestamp_dirs(work_dir: str):
    if not osp.isdir(work_dir):
        return []
    return sorted(
        d for d in os.listdir(work_dir)
        if osp.isdir(osp.join(work_dir, d))
    )


def detect_run_dir(work_dir: str, before_dirs, explicit_reuse):
    if explicit_reuse:
        return osp.join(work_dir, explicit_reuse)
    after_dirs = list_timestamp_dirs(work_dir)
    new_dirs = [d for d in after_dirs if d not in before_dirs]
    if new_dirs:
        return osp.join(work_dir, sorted(new_dirs)[-1])
    if after_dirs:
        return osp.join(work_dir, after_dirs[-1])
    return None


def format_command(cmd):
    return ' '.join(f'"{x}"' if ' ' in x else x for x in cmd)


def run_one(python_exec: str, config_rel_path: str, mode: str, reuse):
    cfg = load_cfg(config_rel_path)
    work_dir = osp.join(ROOT_DIR, cfg['work_dir'])
    before_dirs = list_timestamp_dirs(work_dir)

    cmd = [python_exec, osp.join(ROOT_DIR, 'run.py'), osp.join(ROOT_DIR, config_rel_path), '--mode', mode]
    if reuse:
        cmd.extend(['--reuse', reuse])

    print('\n' + '=' * 88)
    print(f'Starting shard config: {config_rel_path}')
    print('Command:', format_command(cmd))
    print('=' * 88)
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    run_dir = detect_run_dir(work_dir, before_dirs, reuse)
    return {
        'config': config_rel_path,
        'returncode': result.returncode,
        'work_dir': work_dir,
        'run_dir': run_dir,
        'command': format_command(cmd),
    }


def post_process_mtbench101(judgement: str):
    import re
    match = re.search(r'(?:Final Rating|Rating):\s*\[\[?([0-9]+)\]?\]', judgement)
    if not match:
        match = re.search(r'\[([0-9]+)\]', judgement)
    if not match:
        return None
    return int(match.group(1))


def collect_scores_from_result_file(result_file: str):
    import mmengine

    result = mmengine.load(result_file)
    task_multi_id_scores = defaultdict(list)
    invalid = 0
    for _, item in result.items():
        score = post_process_mtbench101(item['prediction'])
        if score is None:
            invalid += 1
            continue
        gold = item['gold']
        task = gold['task']
        multi_id = gold['multi_id']
        task_multi_id_scores[(task, multi_id)].append(score)
    return task_multi_id_scores, len(result), invalid


def get_result_paths(cfg, run_dir):
    model_abbr = cfg['models'][0]['abbr']
    dataset_abbr = cfg['datasets'][0]['abbr']
    if 'meta_judge_model' in cfg:
        judge_abbr = cfg['meta_judge_model']['abbr']
    else:
        judge_abbr = cfg['judge_models'][0]['abbr']
    result_dir = osp.join(run_dir, 'results', f'{model_abbr}_judged-by--{judge_abbr}')
    result_file = osp.join(result_dir, f'{dataset_abbr}.json')
    partial_result_file = osp.join(result_dir, f'{dataset_abbr}_0.json')
    return result_dir, result_file, partial_result_file


def merge_task_scores(run_infos):
    merged_task_multi_id_scores = defaultdict(list)
    shard_rows = []

    for info in run_infos:
        cfg = load_cfg(info['config'])
        run_dir = info['run_dir']
        if not run_dir or not osp.isdir(run_dir):
            shard_rows.append([info['config'], 'missing_run_dir', '', '', ''])
            continue

        _, result_file, partial_result_file = get_result_paths(cfg, run_dir)

        if osp.exists(result_file):
            score_map, total, invalid = collect_scores_from_result_file(result_file)
            source_file = result_file
        elif osp.exists(partial_result_file):
            score_map, total, invalid = collect_scores_from_result_file(partial_result_file)
            source_file = partial_result_file
        else:
            shard_rows.append([info['config'], 'missing_result', run_dir, '', ''])
            continue

        for key, scores in score_map.items():
            merged_task_multi_id_scores[key].extend(scores)
        shard_rows.append([info['config'], 'ok', run_dir, source_file, f'total={total}, invalid={invalid}'])

    task_scores = defaultdict(list)
    for (task, _multi_id), scores in merged_task_multi_id_scores.items():
        task_scores[task].append(min(scores))

    final_task_scores = {
        task: sum(scores) / len(scores) if scores else 0.0
        for task, scores in sorted(task_scores.items())
    }
    return shard_rows, final_task_scores


def write_summary(run_infos, shard_rows, final_task_scores):
    os.makedirs(SUMMARY_ROOT, exist_ok=True)
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_dir = osp.join(SUMMARY_ROOT, time_str)
    os.makedirs(summary_dir, exist_ok=True)

    run_status_csv = osp.join(summary_dir, 'run_status.csv')
    with open(run_status_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'returncode', 'run_dir', 'command'])
        for info in run_infos:
            writer.writerow([info['config'], info['returncode'], info['run_dir'] or '', info['command']])

    shard_status_csv = osp.join(summary_dir, 'shard_result_status.csv')
    with open(shard_status_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'status', 'run_dir', 'result_file', 'note'])
        writer.writerows(shard_rows)

    merged_task_score_csv = osp.join(summary_dir, 'merged_task_score.csv')
    with open(merged_task_score_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task', 'score'])
        for task, score in final_task_scores.items():
            writer.writerow([task, score])

    return summary_dir, run_status_csv, shard_status_csv, merged_task_score_csv


def execute_all(args):
    configs = PART_CONFIGS[:]
    results = []
    max_workers = max(1, min(args.max_parallel, len(configs)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_one, args.python, config_rel_path, args.mode, args.reuse): config_rel_path
            for config_rel_path in configs
        }
        for future in as_completed(future_map):
            info = future.result()
            results.append(info)
            print(f"\n[finished] {info['config']} -> returncode={info['returncode']}, run_dir={info['run_dir']}")
            if args.stop_on_failure and info['returncode'] != 0:
                for pending in future_map:
                    if not pending.done():
                        pending.cancel()
                break

    result_order = {config: idx for idx, config in enumerate(configs)}
    results.sort(key=lambda item: result_order[item['config']])
    return results


def main():
    args = parse_args()
    run_infos = execute_all(args)
    shard_rows, final_task_scores = merge_task_scores(run_infos)
    summary_dir, run_status_csv, shard_status_csv, merged_task_score_csv = write_summary(
        run_infos, shard_rows, final_task_scores)

    print('\n' + '=' * 88)
    print('Shard execution summary')
    for info in run_infos:
        print(f"- {info['config']}: returncode={info['returncode']}, run_dir={info['run_dir']}")
    print('\nMerged task scores:')
    if final_task_scores:
        for task, score in final_task_scores.items():
            print(f'  {task}: {score:.4f}')
    else:
        print('  No merged scores were produced. Check shard_result_status.csv.')
    print('\nSummary directory:')
    print(' ', summary_dir)
    print('Summary files:')
    print(' ', run_status_csv)
    print(' ', shard_status_csv)
    print(' ', merged_task_score_csv)
    print('=' * 88)


if __name__ == '__main__':
    main()
