import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(r'd:\WORKS\综合课设+毕设\opencompass')
CONFIG_DIR = ROOT_DIR / 'configs' / 'mtbench101_plus_task13x30'
SUMMARY_DIR = ROOT_DIR / 'outputs' / 'mtbench101_plus_task13x30' / '_run_summaries'
TARGET_MODEL_ABBR = 'glm-4_6v-flashx'
TASKS = ['AR', 'CC', 'CM', 'CR', 'FR', 'GR', 'IC', 'MR', 'PI', 'SA', 'SC', 'SI', 'TS']


def parse_args():
    parser = argparse.ArgumentParser(description='Run one fixed target model on selected MT-Bench-101 tasks.')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--mode', default='all', choices=['all', 'infer', 'eval', 'viz'])
    parser.add_argument('--max-parallel', type=int, default=13)
    parser.add_argument('--reuse', default=None)
    parser.add_argument('--tasks', nargs='+', default=TASKS, help='Subset of tasks to run, e.g. --tasks AR or --tasks AR CC')
    return parser.parse_args()


def normalize_tasks(tasks):
    normalized = []
    invalid = []
    for task in tasks:
        task = task.upper()
        if task not in TASKS:
            invalid.append(task)
            continue
        if task not in normalized:
            normalized.append(task)
    if invalid:
        raise ValueError(f'Unsupported tasks: {invalid}. Valid tasks: {TASKS}')
    if not normalized:
        raise ValueError('No valid tasks were provided.')
    return normalized


def get_configs(selected_tasks):
    configs = []
    for task in selected_tasks:
        path = CONFIG_DIR / f'eval_mtbench101_plus_task13x30_{task}.py'
        if not path.exists():
            raise FileNotFoundError(f'Missing config: {path}')
        configs.append((task, path))
    return configs


def run_one(python_exec: str, config_path: Path, mode: str, reuse: str | None):
    cmd = [python_exec, str(ROOT_DIR / 'run.py'), str(config_path), '--mode', mode]
    if reuse:
        cmd.extend(['--reuse', reuse])
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    return {
        'task': config_path.stem.split('_')[-1],
        'config': str(config_path),
        'returncode': result.returncode,
        'command': ' '.join(cmd)
    }


def write_summary(args, selected_tasks, results):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    task_tag = 'all' if len(selected_tasks) == len(TASKS) else '-'.join(selected_tasks)
    out_dir = SUMMARY_DIR / f'{TARGET_MODEL_ABBR}_{task_tag}_{time_str}'
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload['tasks'] = selected_tasks
    (out_dir / 'run_args.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    with open(out_dir / 'run_status.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task', 'config', 'returncode', 'command'])
        for item in results:
            writer.writerow([item['task'], item['config'], item['returncode'], item['command']])
    return out_dir


def main():
    args = parse_args()
    selected_tasks = normalize_tasks(args.tasks)
    configs = get_configs(selected_tasks)
    results = []
    max_workers = max(1, min(args.max_parallel, len(configs)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, args.python, path, args.mode, args.reuse) for _, path in configs]
        for future in as_completed(futures):
            item = future.result()
            results.append(item)
            print(f"[finished] task={item['task']} returncode={item['returncode']}")

    results.sort(key=lambda x: TASKS.index(x['task']))
    summary_dir = write_summary(args, selected_tasks, results)
    print('\nSummary:', summary_dir)
    for item in results:
        print(f"- {item['task']}: {item['returncode']}")


if __name__ == '__main__':
    main()
