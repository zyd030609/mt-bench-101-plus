import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(r'd:\WORKS\综合课设+毕设\opencompass')
SRC = ROOT / 'data' / 'subjective' / 'mtbench101.jsonl'
OUT = ROOT / 'data' / 'subjective' / 'mtbench101_task13x30.jsonl'
OUT_DIR = ROOT / 'data' / 'subjective' / 'mtbench101_task13x30_by_task'
META = ROOT / 'data' / 'subjective' / 'mtbench101_task13x30_meta.json'
PER_TASK = 30

buckets = defaultdict(list)
for line in SRC.read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    item = json.loads(line)
    buckets[item['task']].append(item)

selected = []
summary = {}
OUT_DIR.mkdir(parents=True, exist_ok=True)

for task in sorted(buckets):
    chosen = buckets[task][:PER_TASK]
    selected.extend(chosen)
    task_file = OUT_DIR / f'mtbench101_task13x30_{task}.jsonl'
    task_file.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in chosen) + '\n', encoding='utf-8')
    summary[task] = {
        'available': len(buckets[task]),
        'selected': len(chosen),
        'selected_ids': [x['id'] for x in chosen],
        'file': str(task_file),
    }

OUT.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in selected) + '\n', encoding='utf-8')
META.write_text(json.dumps({
    'source': str(SRC),
    'output': str(OUT),
    'per_task': PER_TASK,
    'task_count': len(summary),
    'total_selected': len(selected),
    'tasks': summary,
}, ensure_ascii=False, indent=2), encoding='utf-8')

print(str(OUT))
print(str(META))
print('total_selected=', len(selected))
for task in sorted(summary):
    print(task, summary[task]['selected'], summary[task]['file'])
